"""GUI-free core of the SMART-G playground (see smartg_playground.py for
the interactive front end, and run_from_config.py for the headless one).

Holds everything that doesn't need a display: the SMART-G render functions,
the pointing-error application, the saved-scenario file format
(PlaygroundConfig) and the simulation object that turns a config into
rendered images. Both front ends call PlaygroundSimulation.render() with a
PlaygroundConfig, so a scenario set up in the GUI and saved to a file
renders identically when run headlessly from that file.

Still imports main.py (for the geometry/SMART-G helpers), which in turn
imports Mayavi/Traits - installed, but no display is needed to import them
(see run_from_config.py).
"""

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import cloud_microphysics
import main
import pointing_noise

OUTPUT_DIR = Path(__file__).with_name("smartg_playground_images")
CONFIG_DIR = Path(__file__).with_name("configs")

# Small/fast presets for quick iteration - this is a sandbox, not the
# production renderer (see main.SMARTG_3D_RESOLUTION_PRESETS for that).
RESOLUTION_PRESETS = {
    "Tiny (16x16)": (16, 16),
    "Small (32x32)": (32, 32),
    "Medium (64x64)": (64, 64),
    # Higher-resolution options. SMART-G uses one Sensor per pixel, so run
    # time and memory grow roughly with the pixel count - expect the larger
    # ones to be slow.
    "Large (128x128)": (128, 128),
    "XL (256x256)": (256, 256),
    "XXL (512x512)": (512, 512),
    "Huge (1024x1024)": (1024, 1024),
    "Low 4:3 (256x192)": (256, 192),
    "High 4:3 (1024x768)": (1024, 768),
}
DEFAULT_RESOLUTION = "Small (32x32)"

# Kept safely inside the true horizon (~66 deg for a ~594 km CLOUDCT-
# altitude orbit, see the horizon half-angle formula in
# viewing_direction_km()'s docstring) so the chosen direction always hits
# Earth well short of grazing incidence.
MAX_OFF_NADIR_DEG = 55.0

SINGLE_MODE = "Single satellite"
FORMATION_MODE = "Formation (frozen)"
MODES = (SINGLE_MODE, FORMATION_MODE)

CONFIG_VERSION = 1

# Surface under the cloud field: SMART-G's flat Lambertian ground (the
# original behaviour), or a wind-roughened sea reflecting by the Cox-Munk
# slope distribution (sun glint and Fresnel reflection, polarised) - see
# build_surface().

SURFACE_LAMBERTIAN = "lambertian"
SURFACE_OCEAN = "ocean"
SURFACES = (SURFACE_LAMBERTIAN, SURFACE_OCEAN)
DEFAULT_WIND_SPEED_MS = 5.0
MAX_WIND_SPEED_MS = 30.0

DEFAULT_NOISE_PARAMS = pointing_noise.PRESETS[pointing_noise.DEFAULT_PRESET]

# The 3D cloud: SMART-G's bundled cumulus.dat (the original behaviour), or a
# cloud built from a BOMEX droplet-spectrum dataset (see cloud_microphysics.py
# and cumulus_file.py). The BOMEX cloud is written once to a cumulus.dat-format
# file in GENERATED_CLOUD_DIR and reused while the dataset and the settings
# that shape it (extinction method, crop) stay the same.
CLOUD_BUNDLED = "bundled"
CLOUD_BOMEX = "bomex"
CLOUD_SOURCES = (CLOUD_BUNDLED, CLOUD_BOMEX)
GENERATED_CLOUD_DIR = Path(__file__).with_name("generated_clouds")
GENERATED_CLOUD_VERSION = 1  # bump when the generated file's content changes
EXTINCTION_METHODS = (cloud_microphysics.EXTINCTION_Q2, cloud_microphysics.EXTINCTION_MIE)
# Scattering phase function of the cloud droplets: the Mie phase functions of
# SMART-G's water-cloud table (the original behaviour), or Henyey-Greenstein
# with asymmetry parameter hg_g. Applies to either cloud source.
PHASE_FUNCTIONS = (cloud_microphysics.PHASE_MIE, cloud_microphysics.PHASE_HG)
DEFAULT_HG_G = 0.85
MIN_CROP_CELLS = 16


def viewing_direction_km(
    satellite_position_km: np.ndarray, off_nadir_deg: float, azimuth_deg: float
) -> np.ndarray:
    """Unit vector, from the satellite, pointing `off_nadir_deg` away from
    straight down (nadir, i.e. toward Earth's center) at `azimuth_deg` -
    the same convention as main.zenith_azimuth_deg() (counterclockwise
    from local east, see main.enu_basis_km()), just built the opposite
    way: an angle in, rather than a vector out.

    0 deg off-nadir is straight down regardless of azimuth. The true
    horizon (ray exactly tangent to Earth) is at
    ``degrees(arccos(EARTH_RADIUS_KM / |satellite_position_km|))`` from
    nadir; azimuth_deg is unused there but must still be given."""
    east, north, up = main.enu_basis_km(satellite_position_km)
    down = -up
    theta = np.radians(off_nadir_deg)
    phi = np.radians(azimuth_deg)
    horizontal = east * np.cos(phi) + north * np.sin(phi)
    direction = down * np.cos(theta) + horizontal * np.sin(theta)
    return direction / np.linalg.norm(direction)


def build_surface(
    surface: str = SURFACE_LAMBERTIAN, wind_speed_ms: float = DEFAULT_WIND_SPEED_MS
):
    """The surface under the cloud field.

    SURFACE_LAMBERTIAN is main.py's flat dark ground (albedo
    main.SMARTG_SURFACE_ALBEDO). SURFACE_OCEAN is a Cox-Munk rough sea at
    `wind_speed_ms` (m/s), reflection only (sur=1, brdf=True) with wave
    shadowing: it gives sun glint and angle-dependent Fresnel reflection,
    but no light enters the water, so there is no water-leaving radiance
    (no blue-green sea colour). It applies over the whole domain."""
    from smartg.albedo import AlbedoCst
    from smartg.surface import LambSurface, RoughSurface

    if surface == SURFACE_OCEAN:
        return RoughSurface(wind=wind_speed_ms, sur=1, brdf=True, wave_shadow=True)
    if surface == SURFACE_LAMBERTIAN:
        return LambSurface(alb=AlbedoCst(main.SMARTG_SURFACE_ALBEDO))
    raise ValueError(f"surface must be one of {SURFACES}, not {surface!r}")


@dataclass(frozen=True)
class CloudSource:
    """The 3D cloud a render uses: a cumulus.dat-format file, and the droplet
    phase function ("mie": SMART-G's water-cloud table, or "hg":
    Henyey-Greenstein with asymmetry parameter `g`)."""

    dat_path: Path
    phase_function: str = cloud_microphysics.PHASE_MIE
    g: float = DEFAULT_HG_G

    @property
    def center_height_km(self) -> float:
        """Height of the middle of the cloud layers, which the satellites aim at."""
        return main.cloud_field_center_height_km(self.dat_path)


def bundled_cloud(
    phase_function: str = cloud_microphysics.PHASE_MIE, g: float = DEFAULT_HG_G
) -> CloudSource:
    """SMART-G's bundled cumulus.dat (main.SMARTG_3D_CLOUD_PATH)."""
    return CloudSource(main.SMARTG_3D_CLOUD_PATH, phase_function, g)


def bomex_cloud_file(
    netcdf_path: Path | str,
    extinction: str,
    crop_cells: int = 0,
    on_status: Callable[[str], None] | None = None,
) -> Path:
    """The cumulus.dat-format file for a BOMEX dataset, created in
    GENERATED_CLOUD_DIR if it is not there yet (a full dataset takes about
    half a minute) and reused after that. The file name carries a hash of the
    dataset (path, size, modification time) and of the settings that shape the
    cloud: the extinction method and the crop. `crop_cells` > 0 crops a square
    of that many cells around the column with the largest optical depth, 0
    keeps the whole field."""
    import cumulus_file

    netcdf_path = Path(netcdf_path).expanduser().resolve()
    if not netcdf_path.is_file():
        raise FileNotFoundError(f"BOMEX dataset not found: {netcdf_path}")
    stat = netcdf_path.stat()
    key = hashlib.sha1(
        f"{netcdf_path}|{stat.st_size}|{int(stat.st_mtime)}|{extinction}|{crop_cells}|"
        f"{GENERATED_CLOUD_VERSION}".encode()
    ).hexdigest()[:10]
    crop_label = f"crop{crop_cells}" if crop_cells else "full"
    dat_path = GENERATED_CLOUD_DIR / f"{netcdf_path.stem}_{extinction}_{crop_label}_{key}.dat"
    if dat_path.exists():
        return dat_path

    if on_status:
        on_status(f"Building the cloud file from {netcdf_path.name} (takes a while the first time)...")
    optics = cloud_microphysics.load_cloud_optics(netcdf_path, extinction=extinction)
    crop = cumulus_file.crop_around_thickest_column(optics, crop_cells) if crop_cells else None
    comment = (
        f"{netcdf_path.name}; extinction {extinction} (Q = "
        f"{'2' if extinction == cloud_microphysics.EXTINCTION_Q2 else 'Mie'}); "
        f"w_ref {cloud_microphysics.SMARTG_W_REF_NM:g} nm; "
        f"{crop_label}; made by cumulus_file.py"
    )
    # Write under a temporary name so that a failed run leaves no half-written
    # file for the next one to reuse.
    partial = dat_path.with_name(dat_path.name + ".partial")
    cumulus_file.build_cumulus_file(optics, partial, crop, comment)
    partial.replace(dat_path)
    return dat_path


def resolve_cloud(
    config: "PlaygroundConfig", on_status: Callable[[str], None] | None = None
) -> CloudSource:
    """The cloud a config asks for (building the BOMEX cloud file if needed)."""
    if config.cloud_source == CLOUD_BUNDLED:
        return bundled_cloud(config.phase_function, config.hg_g)
    dat_path = bomex_cloud_file(
        config.bomex_netcdf, config.extinction_method, config.bomex_crop_cells, on_status
    )
    return CloudSource(dat_path, config.phase_function, config.hg_g)


def build_scene_3d(
    lat_deg: float,
    periodic: bool,
    surface: str = SURFACE_LAMBERTIAN,
    wind_speed_ms: float = DEFAULT_WIND_SPEED_MS,
    cloud: CloudSource | None = None,
):
    """Like main.build_smartg_scene_3d(), except the domain's horizontal
    boundary condition (periodic vs. non-periodic + extended margin) is a
    parameter instead of a fixed choice - this sandbox exists specifically
    to compare the two (see the finding that led to main.py fixing on
    non-periodic, in build_smartg_scene_3d()'s own docstring), and the cloud
    (see CloudSource; default: the bundled cumulus field with its own
    water-cloud-table phase function) is a parameter too.

    Returns (wavelength_nm, atmosphere, surface, grid_3d), same as
    main.build_smartg_scene_3d()."""
    from smartg.atmosphere import Atm1D, Atm3D, Cloud3D, read_i3rc_cloud
    from smartg.grid3d import Grid3D

    cloud = cloud or bundled_cloud()
    if not cloud.dat_path.exists():
        raise FileNotFoundError(
            f"{cloud.dat_path} not found; the bundled cumulus.dat is downloaded with "
            "main.ensure_smartg_auxdata(data_type='IPRT')."
        )

    field = read_i3rc_cloud(cloud.dat_path)
    phase = (
        cloud_microphysics.henyey_greenstein_phase(cloud.g)
        if cloud.phase_function == cloud_microphysics.PHASE_HG
        else None  # Cloud3D then uses the phase functions of its water-cloud table
    )
    cloud_3d = Cloud3D(
        main.SMARTG_CLOUD_FNAME,
        w_ref=main.SMARTG_3D_CLOUD_W_REF_NM,
        ds=field,
        reff_acc=main.SMARTG_3D_CLOUD_REFF_ACC,
        reff_min=main.SMARTG_3D_CLOUD_REFF_MIN_UM,
        phase=phase,
    )
    grid_kwargs = {"periodic": periodic}
    if not periodic:
        grid_kwargs["horiz_extend_length"] = main.SMARTG_3D_HORIZ_EXTEND_KM
    grid_3d = Grid3D(
        field["x_bounds"].values,
        field["y_bounds"].values,
        field["z_bounds"].values,
        **grid_kwargs,
    )
    atm_1d = Atm1D(main.SMARTG_AFGL_PROFILE, lat=lat_deg)
    atmosphere = Atm3D(atm_1d=atm_1d, grid_3d=grid_3d, comp_3d=[cloud_3d])
    surface = build_surface(surface, wind_speed_ms)
    return np.array(main.SMARTG_WAVELENGTHS_NM), atmosphere, surface, grid_3d


def footprint_sensors(
    grid_3d,
    width_px: int,
    height_px: int,
    pos_z: float,
    th_deg: float,
    ph_deg: float,
    periodic: bool,
    offset_east_km: float = 0.0,
    offset_north_km: float = 0.0,
    rotation_rad: float = 0.0,
    target_height_km: float | None = None,
):
    """The image's sensor grid: one sensor per pixel over the cloud field,
    all sharing one viewing direction - main.render_formation_images_3d()'s
    layout, but each pixel samples a point (a main.SENSOR_CELL_KM start cell
    kept inside one voxel), not the pixel's whole area - optionally moved by a pointing error: shifted by the ground
    offset of the actual line of sight (domain x = local east, y = local
    north, the same frame as SMART-G's azimuths) and rotated about its
    centre by the yaw error's component about the local vertical.

    Shifted pixels wrap around on a periodic domain; on a non-periodic one
    they may land in the horizontal extension (cloud-free) but not beyond."""
    xgrid = np.linspace(grid_3d.xgrid[0], grid_3d.xgrid[-1], width_px + 1)
    ygrid = np.linspace(grid_3d.ygrid[0], grid_3d.ygrid[-1], height_px + 1)

    # Same pixel order as get_sensors_grid(): row by row, x varying first.
    xx, yy = np.meshgrid(xgrid[:-1] + np.diff(xgrid) / 2.0, ygrid[:-1] + np.diff(ygrid) / 2.0)
    center_x = (xgrid[0] + xgrid[-1]) / 2.0
    center_y = (ygrid[0] + ygrid[-1]) / 2.0
    dx, dy = xx - center_x, yy - center_y
    cos_r, sin_r = np.cos(rotation_rad), np.sin(rotation_rad)
    # Move the footprint so the line of sight passes through the field's
    # centre at the cloud field's centre height (see main.slant_offset_km()).
    drift_east_km, drift_north_km = main.slant_offset_km(th_deg, ph_deg, pos_z, target_height_km)
    xx = center_x + cos_r * dx - sin_r * dy + offset_east_km - drift_east_km
    yy = center_y + sin_r * dx + cos_r * dy + offset_north_km - drift_north_km

    if periodic:
        x0, x1 = grid_3d.xgrid[0], grid_3d.xgrid[-1]
        y0, y1 = grid_3d.ygrid[0], grid_3d.ygrid[-1]
        xx = x0 + (xx - x0) % (x1 - x0)
        yy = y0 + (yy - y0) % (y1 - y0)
    elif (
        xx.min() <= grid_3d.xGRID[0]
        or xx.max() >= grid_3d.xGRID[-1]
        or yy.min() <= grid_3d.yGRID[0]
        or yy.max() >= grid_3d.yGRID[-1]
    ):
        raise ValueError(
            f"the image footprint was moved past the simulation domain's edge "
            f"(+-{grid_3d.xGRID[-1]:.1f} km) by the pointing error "
            f"({offset_east_km:+.2f} km E, {offset_north_km:+.2f} km N) and the "
            f"oblique view ({-drift_east_km:+.2f} km E, {-drift_north_km:+.2f} km N)"
        )

    return main.sensors_at(grid_3d, xx, yy, pos_z, th_deg, ph_deg)


def pointing_offset(
    commanded_point_km: np.ndarray, true_point_km: np.ndarray
) -> tuple[float, float]:
    """(east, north) km from the commanded ground point to the one the
    camera actually looks at, in the commanded point's local frame - the
    cloud field's own (x, y) frame."""
    east, north, _ = main.enu_basis_km(commanded_point_km)
    delta = true_point_km - commanded_point_km
    return float(np.dot(delta, east)), float(np.dot(delta, north))


def render_view(
    satellite_position_km: np.ndarray,
    point_a_km: np.ndarray,
    sza_deg: float,
    saa_deg: float,
    periodic: bool,
    width_px: int,
    height_px: int,
    n_photons: float | None = None,
    true_point_km: np.ndarray | None = None,
    footprint_rotation_rad: float = 0.0,
    surface: str = SURFACE_LAMBERTIAN,
    wind_speed_ms: float = DEFAULT_WIND_SPEED_MS,
    cloud: CloudSource | None = None,
) -> tuple[np.ndarray, float, float]:
    """Run one SMART-G 3D radiative-transfer view of the cloud field,
    centred on point_a_km, as seen from satellite_position_km, under a
    directly-chosen (not ephemeris-derived) sun direction.

    Same technique as main.render_formation_images_3d() for a single
    satellite: backward mode, one Sensor grid at the field's top aimed
    along the satellite's viewing direction, a LocalEstimate toward the
    chosen sun direction.

    Returns (reflectance array of shape (height_px, width_px,
    n_wavelengths), vza_deg, vaa_deg) - the ground-based viewing angles
    at point_a_km this run actually used (see main.viewing_angles_deg()),
    which generally differ from the satellite's own off-nadir angle.

    With a pointing error (see pointing_noise.py), `true_point_km` is where
    the actual line of sight hits the ground: the cloud field stays centred
    on the commanded point_a_km, while the image footprint moves to
    true_point_km (and turns by `footprint_rotation_rad`), viewed along the
    actual line of sight. The returned angles are then the actual ones.

    `surface`/`wind_speed_ms`: see build_surface(). `cloud`: see
    build_scene_3d() (default: the bundled cumulus field)."""
    main.ensure_cuda_compatible_msvc_on_path()
    from smartg.smartg import LocalEstimate, Smartg

    if n_photons is None:
        n_photons = main.SMARTG_3D_PHOTONS_PER_PIXEL * width_px * height_px
    if true_point_km is None:
        true_point_km = point_a_km
    cloud = cloud or bundled_cloud()

    lat_deg = main.geocentric_latitude_deg(point_a_km)
    wavelength_nm, atmosphere, surface, grid_3d = build_scene_3d(
        lat_deg, periodic, surface, wind_speed_ms, cloud
    )
    atmosphere_profile = atmosphere.calc(wavelength_nm)

    pos_z = grid_3d.zGRID[-1] - main.SMARTG_3D_SENSOR_TOP_OFFSET_KM

    vza_deg, vaa_deg = main.viewing_angles_deg(
        satellite_position_km, main.aim_point_km(true_point_km, cloud.center_height_km)
    )
    offset_east_km, offset_north_km = pointing_offset(point_a_km, true_point_km)
    sensors = footprint_sensors(
        grid_3d,
        width_px,
        height_px,
        pos_z,
        th_deg=180.0 - vza_deg,
        ph_deg=(vaa_deg + 180.0) % 360.0,
        periodic=periodic,
        offset_east_km=offset_east_km,
        offset_north_km=offset_north_km,
        rotation_rad=footprint_rotation_rad,
        target_height_km=cloud.center_height_km,
    )
    le = LocalEstimate(
        th_deg=np.array([sza_deg]),
        phi_deg=np.array([saa_deg]),
        count_level=np.array([0]),  # 0 = UPTOA, see smartg.sensor.Sensor
    )

    ds = Smartg(back=True, opt3d=True).run(
        wavelength=wavelength_nm,
        atmosphere=atmosphere_profile,
        surface=surface,
        sensor=sensors,
        le=le,
        n_photons=n_photons,
        progress=False,
    )
    toa_reflectance = ds["I_up (TOA)"].isel(**{"Azimuth angles": 0, "Zenith angles": 0})
    reflectance = toa_reflectance.values.reshape(height_px, width_px, len(wavelength_nm))
    return reflectance, vza_deg, vaa_deg


def render_formation_frozen(
    target_km: np.ndarray,
    positions_km: np.ndarray,
    satellite_indices: list[int],
    sza_deg: float,
    saa_deg: float,
    periodic: bool,
    width_px: int,
    height_px: int,
    n_photons: float | None = None,
    pointing: dict[int, tuple[np.ndarray, float]] | None = None,
    surface: str = SURFACE_LAMBERTIAN,
    wind_speed_ms: float = DEFAULT_WIND_SPEED_MS,
    cloud: CloudSource | None = None,
) -> dict[int, np.ndarray]:
    """Render the CloudCT formation's frozen-in-time geometry, exactly as
    main.py defines it (main.formation_positions_km(): satellite index
    main.NADIR_REFERENCE_INDEX looks at its own nadir, every other
    satellite looks at that same shared ground target - see
    main.render_formation_images_3d()), under a directly-chosen sun
    direction and a togglable periodic/non-periodic domain.

    Unlike main.render_formation_images_3d(), which always derives the
    sun direction from real ephemeris and always uses a non-periodic
    domain, both are free parameters here - the point of this sandbox.
    Otherwise the same technique: one shared atmosphere, built and
    `.calc()`-ed once, and one `Smartg` instance reused across a
    sequential per-satellite `run()` loop (see that function's own
    docstring for why - GPU memory scales with sensor count, so the
    satellites aren't batched into a single combined run).

    `positions_km` are the formation's satellite positions at the single
    frozen moment being rendered (see main.formation_positions_km());
    `target_km` is the shared nadir target their images are centred on
    (see main.nadir_target_km()).

    `pointing` optionally gives, per satellite index, (true ground point
    km, footprint rotation rad) for a pointing error - same meaning as
    render_view()'s `true_point_km`/`footprint_rotation_rad`; satellites
    missing from it point exactly at the target.

    Returns {satellite_index: (height_px, width_px, n_wavelengths)
    reflectance array}, for each index in `satellite_indices`."""
    main.ensure_cuda_compatible_msvc_on_path()
    from smartg.smartg import LocalEstimate, Smartg

    if n_photons is None:
        n_photons = main.SMARTG_3D_PHOTONS_PER_PIXEL * width_px * height_px
    pointing = pointing or {}
    cloud = cloud or bundled_cloud()

    lat_deg = main.geocentric_latitude_deg(target_km)
    wavelength_nm, atmosphere, surface, grid_3d = build_scene_3d(
        lat_deg, periodic, surface, wind_speed_ms, cloud
    )
    atmosphere_profile = atmosphere.calc(wavelength_nm)

    pos_z = grid_3d.zGRID[-1] - main.SMARTG_3D_SENSOR_TOP_OFFSET_KM

    le = LocalEstimate(
        th_deg=np.array([sza_deg]),
        phi_deg=np.array([saa_deg]),
        count_level=np.array([0]),  # 0 = UPTOA, see smartg.sensor.Sensor
    )

    sg = Smartg(back=True, opt3d=True)

    images = {}
    for index in satellite_indices:
        true_point_km, rotation_rad = pointing.get(index, (target_km, 0.0))
        vza_deg, vaa_deg = main.viewing_angles_deg(
            positions_km[index], main.aim_point_km(true_point_km, cloud.center_height_km)
        )
        offset_east_km, offset_north_km = pointing_offset(target_km, true_point_km)
        sensors = footprint_sensors(
            grid_3d,
            width_px,
            height_px,
            pos_z,
            th_deg=180.0 - vza_deg,
            ph_deg=(vaa_deg + 180.0) % 360.0,
            periodic=periodic,
            offset_east_km=offset_east_km,
            offset_north_km=offset_north_km,
            rotation_rad=rotation_rad,
            target_height_km=cloud.center_height_km,
        )
        ds = sg.run(
            wavelength=wavelength_nm,
            atmosphere=atmosphere_profile,
            surface=surface,
            sensor=sensors,
            le=le,
            n_photons=n_photons,
            progress=False,
        )
        toa_reflectance = ds["I_up (TOA)"].isel(**{"Azimuth angles": 0, "Zenith angles": 0})
        images[index] = toa_reflectance.values.reshape(height_px, width_px, len(wavelength_nm))

    return images


@dataclass
class PlaygroundConfig:
    """Every setting of one playground scenario - the GUI's controls, plus
    the moment in time and the orbital elements the satellite geometry was
    computed from (the GUI places satellites at "now" from a live TLE, so a
    later headless run needs those pinned down or it would silently render
    a different geometry)."""

    tle: str  # TLE text (name line + 2 element lines) the orbit comes from
    time_utc: str  # ISO 8601 UTC moment all positions/frozen formation are at

    mode: str = SINGLE_MODE
    off_nadir_deg: float = 0.0
    view_azimuth_deg: float = 0.0
    satellites_to_render: list[int] = field(
        default_factory=lambda: [main.NADIR_REFERENCE_INDEX]
    )
    sza_deg: float = 30.0
    saa_deg: float = 90.0
    periodic: bool = False
    surface: str = SURFACE_LAMBERTIAN
    wind_speed_ms: float = DEFAULT_WIND_SPEED_MS  # only used by the ocean surface
    resolution: str = DEFAULT_RESOLUTION
    photons_per_pixel: float = main.SMARTG_3D_PHOTONS_PER_PIXEL
    # Brightness gain of the saved PNGs: (gain * reflectance) ** (1/1.8),
    # clipped to [0, 1] - see main.reflectance_to_image(). The raw reflectance
    # is saved next to each PNG (.npy), unaffected by it.
    display_gain: float = main.SMARTG_3D_DEFAULT_GAIN

    # The 3D cloud (see CLOUD_SOURCES). With "bomex", the cloud is built from
    # the droplet spectra of the netCDF dataset `bomex_netcdf`: beta (the
    # extinction coefficient) from Eq. 3 with Q = 2 ("q2") or the Mie Q of each
    # radius bin ("mie"), optionally cropped to a square of `bomex_crop_cells`
    # cells around the thickest column (0 = the whole field).
    cloud_source: str = CLOUD_BUNDLED
    bomex_netcdf: str = ""
    extinction_method: str = cloud_microphysics.EXTINCTION_Q2
    bomex_crop_cells: int = 0
    # Droplet phase function (either cloud source) and, for "hg", its asymmetry
    # parameter.
    phase_function: str = cloud_microphysics.PHASE_MIE
    hg_g: float = DEFAULT_HG_G

    pointing_noise_enabled: bool = False
    sigma_bias_deg: float = DEFAULT_NOISE_PARAMS["sigma_bias_deg"]
    sigma_acquisition_deg: float = DEFAULT_NOISE_PARAMS["sigma_acquisition_deg"]
    sigma_drift_deg: float = DEFAULT_NOISE_PARAMS["sigma_drift_deg"]
    drift_tau_s: float = DEFAULT_NOISE_PARAMS["drift_tau_s"]
    sigma_jitter_deg: float = DEFAULT_NOISE_PARAMS["sigma_jitter_deg"]
    yaw_factor: float = pointing_noise.DEFAULT_YAW_FACTOR
    acquisition_interval_s: float = pointing_noise.DEFAULT_ACQUISITION_INTERVAL_S
    noise_seed: int = 0

    # Headless runs only: how many renders (acquisitions) to do. Each one
    # draws fresh pointing errors, so it only means something with the
    # noise enabled.
    renders: int = 1

    def noise_params(self) -> pointing_noise.PointingNoiseParams:
        return pointing_noise.PointingNoiseParams(
            sigma_bias_deg=self.sigma_bias_deg,
            sigma_acquisition_deg=self.sigma_acquisition_deg,
            sigma_drift_deg=self.sigma_drift_deg,
            drift_tau_s=self.drift_tau_s,
            sigma_jitter_deg=self.sigma_jitter_deg,
            yaw_factor=self.yaw_factor,
            acquisition_interval_s=self.acquisition_interval_s,
        )

    def validate(self) -> None:
        """Raise ValueError for a setting no front end could have produced,
        e.g. a hand-edited config file."""
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not {self.mode!r}")
        if self.resolution not in RESOLUTION_PRESETS:
            raise ValueError(
                f"resolution must be one of {list(RESOLUTION_PRESETS)}, "
                f"not {self.resolution!r}"
            )
        if self.surface not in SURFACES:
            raise ValueError(f"surface must be one of {SURFACES}, not {self.surface!r}")
        if not 0.0 <= self.wind_speed_ms <= MAX_WIND_SPEED_MS:
            raise ValueError(f"wind_speed_ms must be in [0, {MAX_WIND_SPEED_MS}]")
        if not 0.0 <= self.off_nadir_deg <= MAX_OFF_NADIR_DEG:
            raise ValueError(f"off_nadir_deg must be in [0, {MAX_OFF_NADIR_DEG}]")
        if not all(0 <= i < main.NUM_SATELLITES for i in self.satellites_to_render):
            raise ValueError(
                f"satellites_to_render must be indices in [0, {main.NUM_SATELLITES - 1}]"
            )
        if self.mode == FORMATION_MODE and not self.satellites_to_render:
            raise ValueError("satellites_to_render is empty")
        if not self.display_gain > 0:
            raise ValueError("display_gain must be > 0")
        if self.photons_per_pixel < 1:
            raise ValueError("photons_per_pixel must be >= 1")
        if self.renders < 1:
            raise ValueError("renders must be >= 1")
        if self.cloud_source not in CLOUD_SOURCES:
            raise ValueError(f"cloud_source must be one of {CLOUD_SOURCES}, not {self.cloud_source!r}")
        if self.cloud_source == CLOUD_BOMEX and not self.bomex_netcdf.strip():
            raise ValueError("bomex_netcdf (the dataset's path) is required for the bomex cloud")
        if self.extinction_method not in EXTINCTION_METHODS:
            raise ValueError(
                f"extinction_method must be one of {EXTINCTION_METHODS}, not {self.extinction_method!r}"
            )
        if self.bomex_crop_cells != 0 and self.bomex_crop_cells < MIN_CROP_CELLS:
            raise ValueError(f"bomex_crop_cells must be 0 (no crop) or >= {MIN_CROP_CELLS}")
        if self.phase_function not in PHASE_FUNCTIONS:
            raise ValueError(
                f"phase_function must be one of {PHASE_FUNCTIONS}, not {self.phase_function!r}"
            )
        if not -1.0 < self.hg_g < 1.0:
            raise ValueError("hg_g must be in (-1, 1)")
        if self.drift_tau_s <= 0:
            raise ValueError("drift_tau_s must be > 0")
        parse_time_utc(self.time_utc)  # raises ValueError if unparseable

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"version": CONFIG_VERSION, **asdict(self)}
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "PlaygroundConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        version = data.pop("version", None)
        if version != CONFIG_VERSION:
            raise ValueError(
                f"unsupported config version {version!r} (expected {CONFIG_VERSION})"
            )
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown config setting(s): {', '.join(unknown)}")
        missing = [name for name in ("tle", "time_utc") if name not in data]
        if missing:
            raise ValueError(f"config is missing required setting(s): {', '.join(missing)}")
        config = cls(**data)
        config.validate()
        return config


def parse_time_utc(time_utc: str) -> datetime:
    parsed = datetime.fromisoformat(time_utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


class PlaygroundSimulation:
    """The satellite/formation geometry frozen at one moment, plus the
    pointing-error random state - everything needed to turn a
    PlaygroundConfig into rendered images."""

    def __init__(self, satellite, t, tle_text: str, noise_seed: int = 0) -> None:
        self.satellite = satellite
        self.t = t
        self.tle_text = tle_text
        self.position_km = satellite.at(t).frame_xyz(main.itrs).km

        # The formation's positions, frozen at this same single moment t -
        # see main.formation_positions_km()/main.render_formation_images_3d().
        self.formation_time_offsets_seconds = main.formation_time_offsets_seconds(
            satellite, t
        )
        self.formation_positions_km = main.formation_positions_km(
            satellite, t, self.formation_time_offsets_seconds
        )
        self.formation_target_km = main.nadir_target_km(
            self.formation_positions_km[main.NADIR_REFERENCE_INDEX]
        )

        # Along-track direction for each satellite's camera frame (roll
        # axis): the orbit normal crossed with its position. Every
        # satellite shares the orbital plane, so one normal serves all.
        self._orbit_normal = main.orbital_normal_itrs(satellite, t)
        self.noise = pointing_noise.PointingNoise(noise_seed)

    @classmethod
    def from_config(cls, config: PlaygroundConfig) -> "PlaygroundSimulation":
        from skyfield.api import load

        satellite = main.build_satellite(config.tle)
        t = load.timescale().from_datetime(parse_time_utc(config.time_utc))
        return cls(satellite, t, config.tle, config.noise_seed)

    @property
    def time_utc(self) -> str:
        return self.t.utc_datetime().isoformat()

    def reset_noise(self, seed: int) -> None:
        self.noise.reset(seed)

    def point_a_km(self, off_nadir_deg: float, view_azimuth_deg: float) -> np.ndarray:
        """Single-satellite mode's ground point: where the chosen viewing
        ray from the satellite hits Earth's surface."""
        direction = viewing_direction_km(
            self.position_km, off_nadir_deg, view_azimuth_deg
        )
        return main.ray_sphere_intersection_km(
            self.position_km, direction, main.EARTH_RADIUS_KM
        )

    def render(
        self,
        config: PlaygroundConfig,
        on_status: Callable[[str], None] | None = None,
    ) -> str:
        """Run one acquisition of the configured scenario, save its
        image(s) under OUTPUT_DIR, and return a one-line (single) or
        multi-line (formation) description of what was done.

        With the pointing error enabled, this starts a new acquisition (see
        PointingNoise.begin_acquisition()), so successive calls walk the
        same seeded random sequence. `on_status` is called with a progress
        message just before the (slow) SMART-G run starts."""
        config.validate()
        width_px, height_px = RESOLUTION_PRESETS[config.resolution]
        cloud = resolve_cloud(config, on_status)  # may build the BOMEX cloud file first
        if config.pointing_noise_enabled:
            self.noise.begin_acquisition()
        if config.mode == SINGLE_MODE:
            return self._render_single(config, width_px, height_px, on_status, cloud)
        return self._render_formation(config, width_px, height_px, on_status, cloud)

    def pixel_voxel_views(
        self,
        config: PlaygroundConfig,
        on_status: Callable[[str], None] | None = None,
        resolutions: list[str] | None = None,
    ) -> list[dict]:
        """pixel_voxel_statistics() for each view of the configured scenario
        (the single satellite's, or each satellite to render in formation
        mode) at each of the named `resolutions` (default: the config's own).
        Nothing is rendered; the pointing error is not applied. Uses the cloud
        the config asks for (building the BOMEX cloud file first if needed),
        read once. Returns one dict per (resolution, view): `resolution`,
        `width_px`, `height_px`, `label`, `vza_deg` and `stats`."""
        from smartg.atmosphere import read_i3rc_cloud

        config.validate()
        resolutions = resolutions or [config.resolution]
        unknown = [name for name in resolutions if name not in RESOLUTION_PRESETS]
        if unknown:
            raise ValueError(f"unknown resolution(s) {unknown}; choose from {list(RESOLUTION_PRESETS)}")
        cloud = resolve_cloud(config, on_status)
        field = read_i3rc_cloud(cloud.dat_path)
        cloudy = field["ext"].values > 0
        height_km = cloud.center_height_km

        if config.mode == SINGLE_MODE:
            point_a_km = self.point_a_km(config.off_nadir_deg, config.view_azimuth_deg)
            views = [("Single satellite", self.position_km, point_a_km)]
        else:
            views = [
                (f"Satellite {index}", self.formation_positions_km[index], self.formation_target_km)
                for index in sorted(config.satellites_to_render)
            ]

        results = []
        for name in resolutions:
            width_px, height_px = RESOLUTION_PRESETS[name]
            for label, position_km, target_km in views:
                vza_deg, vaa_deg = main.viewing_angles_deg(
                    position_km, main.aim_point_km(target_km, height_km)
                )
                stats = pixel_voxel_statistics(
                    field["x_bounds"].values,
                    field["y_bounds"].values,
                    field["z_bounds"].values,
                    cloudy,
                    width_px,
                    height_px,
                    vza_deg,
                    vaa_deg,
                    height_km,
                    periodic=config.periodic,
                )
                results.append(
                    dict(resolution=name, width_px=width_px, height_px=height_px,
                         label=label, vza_deg=vza_deg, stats=stats)
                )
        return results

    def estimate_pixel_voxels(
        self,
        config: PlaygroundConfig,
        on_status: Callable[[str], None] | None = None,
    ) -> tuple[str, str]:
        """How many voxels a pixel of the configured scenario takes in, at the
        config's resolution (see pixel_voxel_views()). Returns (summary,
        details): one line per view, and the full report. Use it to pick a
        resolution: when a pixel's footprint is much smaller than a voxel,
        neighbouring pixels see the same voxels and a lower resolution gives
        the same information."""
        views = self.pixel_voxel_views(config, on_status)
        summaries = [
            format_pixel_voxel_summary(v["label"], v["stats"], v["vza_deg"], v["width_px"], v["height_px"])
            for v in views
        ]
        reports = [
            format_pixel_voxel_report(v["label"], v["stats"], v["vza_deg"], v["width_px"], v["height_px"])
            for v in views
        ]
        return "\n".join(summaries), "\n".join(reports)

    def _draw_pointing(
        self,
        key,
        position_km: np.ndarray,
        commanded_point_km: np.ndarray,
        params: pointing_noise.PointingNoiseParams,
    ):
        """Draw this acquisition's pointing error for one satellite and
        ray-trace it: returns (true ground point km, footprint rotation rad,
        short summary text)."""
        boresight = commanded_point_km - position_km
        boresight /= np.linalg.norm(boresight)
        along_track = np.cross(self._orbit_normal, position_km)
        error_rad = self.noise.draw_error_rad(key, params)
        true_direction = pointing_noise.perturbed_boresight(boresight, along_track, error_rad)
        true_point_km = main.ray_sphere_intersection_km(
            position_km, true_direction, main.EARTH_RADIUS_KM
        )
        # The yaw error turns the footprint about the boresight; on the
        # ground that's its component about the local vertical (exact at
        # nadir, an approximation off-nadir).
        _, _, up = main.enu_basis_km(commanded_point_km)
        rotation_rad = float(error_rad[2] * np.dot(boresight, up))

        los_error_deg = np.degrees(
            np.arccos(np.clip(np.dot(boresight, true_direction), -1.0, 1.0))
        )
        east_km, north_km = pointing_offset(commanded_point_km, true_point_km)
        summary = (
            f"LOS error {los_error_deg:.4f} deg, ground offset "
            f"{np.hypot(east_km, north_km):.2f} km (E {east_km:+.2f}, N {north_km:+.2f}), "
            f"yaw {np.degrees(error_rad[2]):+.4f} deg"
        )
        return true_point_km, rotation_rad, summary

    def _surface_file_suffix(self, config: PlaygroundConfig) -> str:
        if config.surface == SURFACE_OCEAN:
            return f"_ocean{config.wind_speed_ms:g}ms"
        return ""

    def _cloud_file_suffix(self, config: PlaygroundConfig) -> str:
        """Tells apart the images of different clouds and phase functions; empty
        for the original settings (bundled cloud, water-cloud-table phase)."""
        suffix = ""
        if config.cloud_source == CLOUD_BOMEX:
            crop = f"_crop{config.bomex_crop_cells}" if config.bomex_crop_cells else ""
            suffix += f"_bomex_{config.extinction_method}{crop}"
        if config.phase_function == cloud_microphysics.PHASE_HG:
            suffix += f"_hg{config.hg_g:g}"
        return suffix

    def _noise_file_suffix(self, config: PlaygroundConfig) -> str:
        if not config.pointing_noise_enabled:
            return ""
        return f"_noise_seed{config.noise_seed}_render{self.noise.acquisition}"

    def _render_single(
        self, config: PlaygroundConfig, width_px: int, height_px: int, on_status, cloud: CloudSource
    ) -> str:
        point_a_km = self.point_a_km(config.off_nadir_deg, config.view_azimuth_deg)
        if on_status:
            on_status(
                f"Rendering at {width_px}x{height_px} "
                f"(periodic={config.periodic}, off-nadir={config.off_nadir_deg:.1f} deg, "
                f"view az={config.view_azimuth_deg:.1f} deg)..."
            )

        true_point_km, rotation_rad, pointing_summary = point_a_km, 0.0, ""
        if config.pointing_noise_enabled:
            true_point_km, rotation_rad, pointing_summary = self._draw_pointing(
                "single", self.position_km, point_a_km, config.noise_params()
            )
            pointing_summary += "  |  "

        reflectance, vza_deg, vaa_deg = render_view(
            self.position_km,
            point_a_km,
            config.sza_deg,
            config.saa_deg,
            config.periodic,
            width_px,
            height_px,
            n_photons=config.photons_per_pixel * width_px * height_px,
            true_point_km=true_point_km,
            footprint_rotation_rad=rotation_rad,
            surface=config.surface,
            wind_speed_ms=config.wind_speed_ms,
            cloud=cloud,
        )
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / (
            f"single_offnadir{config.off_nadir_deg:.0f}_az{config.view_azimuth_deg:.0f}_"
            f"sza{config.sza_deg:.0f}_saa{config.saa_deg:.0f}_"
            f"{'periodic' if config.periodic else 'nonperiodic'}"
            f"{self._surface_file_suffix(config)}"
            f"{self._cloud_file_suffix(config)}"
            f"{self._noise_file_suffix(config)}.png"
        )
        gain = config.display_gain
        main.reflectance_to_image(reflectance, gain=gain, gamma=1.8).save(path)
        np.save(path.with_suffix(".npy"), reflectance)  # raw, un-clipped
        figure_path = path.with_name(f"{path.stem}_raw.png")
        save_reflectance_figure(reflectance, figure_path, path.stem, gain)
        return (
            f"{pointing_summary}"
            f"Ground VZA={vza_deg:.1f} VAA={vaa_deg:.1f} deg at point a  |  "
            f"saved {path} (+ .npy, {figure_path.name})\n"
            f"Raw reflectance:\n{reflectance_stats(reflectance, gain)}"
        )

    def _render_formation(
        self, config: PlaygroundConfig, width_px: int, height_px: int, on_status, cloud: CloudSource
    ) -> str:
        satellite_indices = sorted(config.satellites_to_render)
        if on_status:
            on_status(
                f"Rendering {len(satellite_indices)} satellite(s) at {width_px}x{height_px} "
                f"(periodic={config.periodic})..."
            )

        pointing = {}
        summaries = []
        if config.pointing_noise_enabled:
            params = config.noise_params()
            for index in satellite_indices:
                true_point_km, rotation_rad, summary = self._draw_pointing(
                    index,
                    self.formation_positions_km[index],
                    self.formation_target_km,
                    params,
                )
                pointing[index] = (true_point_km, rotation_rad)
                summaries.append(f"Sat {index}: {summary}")

        images = render_formation_frozen(
            self.formation_target_km,
            self.formation_positions_km,
            satellite_indices,
            config.sza_deg,
            config.saa_deg,
            config.periodic,
            width_px,
            height_px,
            n_photons=config.photons_per_pixel * width_px * height_px,
            pointing=pointing,
            surface=config.surface,
            wind_speed_ms=config.wind_speed_ms,
            cloud=cloud,
        )
        subdir = OUTPUT_DIR / (
            f"formation_sza{config.sza_deg:.0f}_saa{config.saa_deg:.0f}_"
            f"{'periodic' if config.periodic else 'nonperiodic'}"
            f"{self._surface_file_suffix(config)}"
            f"{self._cloud_file_suffix(config)}"
            f"{self._noise_file_suffix(config)}"
        )
        subdir.mkdir(parents=True, exist_ok=True)
        gain = config.display_gain
        for index in sorted(images):
            path = subdir / f"satellite_{index:02d}.png"
            main.reflectance_to_image(images[index], gain=gain, gamma=1.8).save(path)
            np.save(path.with_suffix(".npy"), images[index])  # raw, un-clipped
            save_reflectance_figure(
                images[index],
                path.with_name(f"{path.stem}_raw.png"),
                f"{subdir.name} / satellite {index}",
                gain,
            )
            summaries.append(
                f"Sat {index} raw reflectance:\n{reflectance_stats(images[index], gain)}"
            )
        return "\n".join([f"Saved {len(images)} image(s) to {subdir}", *summaries])


def pixel_voxel_statistics(
    x_bounds_km: np.ndarray,
    y_bounds_km: np.ndarray,
    z_bounds_km: np.ndarray,
    cloudy: np.ndarray,
    width_px: int,
    height_px: int,
    vza_deg: float,
    vaa_deg: float,
    target_height_km: float,
    periodic: bool = False,
    n_samples: int = 800,
    seed: int = 0,
) -> dict[str, float]:
    """How many voxels one pixel of an image takes in, for the geometry of
    footprint_sensors(): the pixels are points on a regular grid spanning the
    cloud field (pitch = field size / pixels), all looking along the same
    direction (view zenith `vza_deg`, view azimuth `vaa_deg`), the lines of
    sight crossing the field's centre at `target_height_km`.

    `cloudy` is the (nz, ny, nx) mask of voxels that hold cloud; the bounds are
    those of the cloud file. Two measures, both over the cloud layers (the
    layers from the lowest to the highest cloudy voxel):

    - `voxels_per_pixel`: the pixel's footprint area divided by the area
      a voxel casts on the plane across the line of sight (a box seen along
      direction d casts dx*dy*|dz_d| + dy*dz*|dx_d| + dx*dz*|dy_d|). Below 1
      the pixels are finer than the voxels, so neighbours see the same ones.
    - the voxels that the line of sight of sampled pixels (marched through the
      grid) crosses, and the share of them that the neighbouring pixel's line
      of sight (right and up) crosses too.

    Only the geometric line of sight is counted: photons that scatter reach
    more voxels than the line of sight does."""
    xb, yb, zb = (np.asarray(a, dtype=float) for a in (x_bounds_km, y_bounds_km, z_bounds_km))
    nz, ny, nx = cloudy.shape
    layers = np.flatnonzero(cloudy.any(axis=(1, 2)))
    if layers.size == 0:
        raise ValueError("the cloud has no cloudy voxel")
    k_lo, k_hi = int(layers[0]), int(layers[-1])  # cloud layers, inclusive
    z_lo, z_hi = float(zb[k_lo]), float(zb[k_hi + 1])

    theta = np.radians(vza_deg)
    travel_az = np.radians((vaa_deg + 180.0) % 360.0)  # direction of travel, as in the sensors
    # Horizontal shift of the line of sight per km it descends, as in slant_offset_km().
    shift_x, shift_y = np.tan(theta) * np.cos(travel_az), np.tan(theta) * np.sin(travel_az)

    x0, x1, y0, y1 = xb[0], xb[-1], yb[0], yb[-1]
    pitch_x, pitch_y = (x1 - x0) / width_px, (y1 - y0) / height_px
    cell_x = np.diff(xb).mean()
    cell_y = np.diff(yb).mean()
    cell_z = np.diff(zb[k_lo : k_hi + 2]).mean()

    direction = np.array([np.sin(theta) * np.cos(travel_az), np.sin(theta) * np.sin(travel_az), np.cos(theta)])
    voxel_area = (
        cell_x * cell_y * abs(direction[2])
        + cell_y * cell_z * abs(direction[0])
        + cell_x * cell_z * abs(direction[1])
    )
    pixel_area = pitch_x * pitch_y * np.cos(theta)
    ratio = pixel_area / voxel_area

    # March sample pixels' lines of sight through the cloud layers.
    step_km = 0.25 * min(np.diff(xb).min(), np.diff(yb).min(), np.diff(zb[k_lo : k_hi + 2]).min())
    heights = np.arange(z_hi - 1e-9, z_lo, -step_km * np.cos(theta))
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0

    def voxel_ids(i: int, j: int) -> tuple[np.ndarray, np.ndarray]:
        """(all voxels crossed, cloudy voxels crossed) by pixel column i, row j."""
        px = x0 + (i + 0.5) * pitch_x - cx
        py = y0 + (j + 0.5) * pitch_y - cy
        x = cx + px + (target_height_km - heights) * shift_x
        y = cy + py + (target_height_km - heights) * shift_y
        if periodic:
            x = x0 + (x - x0) % (x1 - x0)
            y = y0 + (y - y0) % (y1 - y0)
        ix = np.searchsorted(xb, x, side="right") - 1
        iy = np.searchsorted(yb, y, side="right") - 1
        iz = np.searchsorted(zb, heights, side="right") - 1
        inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        ix, iy, iz = ix[inside], iy[inside], iz[inside]
        ids = np.unique((iz * ny + iy) * nx + ix)
        cloudy_flat = cloudy.reshape(-1)
        return ids, ids[cloudy_flat[ids]]

    rng = np.random.default_rng(seed)
    columns = rng.integers(0, max(width_px - 1, 1), n_samples)
    rows = rng.integers(0, max(height_px - 1, 1), n_samples)
    crossed, crossed_cloudy, shared, shared_cloudy = [], [], [], []
    for i, j in zip(columns, rows):
        own_all, own_cloudy = voxel_ids(int(i), int(j))
        if own_cloudy.size == 0:
            continue  # a line of sight that misses the cloud entirely
        crossed.append(own_all.size)
        crossed_cloudy.append(own_cloudy.size)
        for di, dj in ((1, 0), (0, 1)):
            if i + di >= width_px or j + dj >= height_px:
                continue
            other_all, other_cloudy = voxel_ids(int(i) + di, int(j) + dj)
            shared.append(np.intersect1d(own_all, other_all).size / own_all.size)
            shared_cloudy.append(np.intersect1d(own_cloudy, other_cloudy).size / own_cloudy.size)

    return {
        "pitch_x_km": pitch_x,
        "pitch_y_km": pitch_y,
        "voxel_km": (cell_x, cell_y, cell_z),
        "voxels_per_pixel": ratio,
        "crossed_voxels": float(np.mean(crossed)) if crossed else 0.0,
        "crossed_cloudy_voxels": float(np.mean(crossed_cloudy)) if crossed_cloudy else 0.0,
        "shared_fraction": float(np.mean(shared)) if shared else 0.0,
        "shared_cloudy_fraction": float(np.mean(shared_cloudy)) if shared_cloudy else 0.0,
        "samples": len(crossed),
        "layers_km": (z_lo, z_hi),
        "matched_width_px": width_px * float(np.sqrt(ratio)),
        "matched_height_px": height_px * float(np.sqrt(ratio)),
    }


def format_pixel_voxel_report(
    label: str, stats: dict[str, float], vza_deg: float, width_px: int, height_px: int
) -> str:
    cx, cy, cz = stats["voxel_km"]
    ratio = stats["voxels_per_pixel"]
    if ratio < 0.9:
        verdict = (
            f"finer than the voxels: neighbouring pixels see the same ones. About "
            f"{stats['matched_width_px']:.0f} x {stats['matched_height_px']:.0f} pixels "
            f"would match the voxel size."
        )
    elif ratio > 1.1:
        verdict = (
            f"coarser than the voxels: each pixel covers several voxels but samples one line of "
            f"sight. About {stats['matched_width_px']:.0f} x {stats['matched_height_px']:.0f} pixels "
            f"would match the voxel size."
        )
    else:
        verdict = "about the size of a voxel's projection."
    return (
        f"{label} (view zenith {vza_deg:.1f} deg, {width_px}x{height_px} pixels):\n"
        f"  pixel pitch {stats['pitch_x_km'] * 1000:.0f} x {stats['pitch_y_km'] * 1000:.0f} m; "
        f"voxel {cx * 1000:.0f} x {cy * 1000:.0f} x {cz * 1000:.0f} m in the cloud layers "
        f"({stats['layers_km'][0]:.2f}-{stats['layers_km'][1]:.2f} km)\n"
        f"  voxels in a pixel's footprint: {ratio:.2f} (pixel footprint / voxel projection) - {verdict}\n"
        f"  a line of sight crosses {stats['crossed_voxels']:.0f} voxels in the cloud layers, "
        f"{stats['crossed_cloudy_voxels']:.0f} of them cloudy; its right/upper neighbour crosses "
        f"{stats['shared_fraction']:.0%} of the same voxels ({stats['shared_cloudy_fraction']:.0%} of the "
        f"cloudy ones), {stats['samples']} lines of sight sampled"
    )


def format_pixel_voxel_summary(
    label: str, stats: dict[str, float], vza_deg: float, width_px: int, height_px: int
) -> str:
    """One line per view: format_pixel_voxel_report() without the details,
    short enough for several satellites to fit in the GUI's status line."""
    return (
        f"{label} (zenith {vza_deg:.0f} deg): pixel {stats['pitch_x_km'] * 1000:.0f} m, "
        f"voxel {stats['voxel_km'][0] * 1000:.0f} m -> {stats['voxels_per_pixel']:.2f} voxels per pixel "
        f"(about {stats['matched_width_px']:.0f}x{stats['matched_height_px']:.0f} px match the voxels); "
        f"neighbours share {stats['shared_fraction']:.0%} of the voxels along the line of sight"
    )


def reflectance_stats(reflectance: np.ndarray, gain: float, indent: str = "  ") -> str:
    """One line per wavelength band describing the raw reflectance (min,
    median, 99th percentile, max) and the share of pixels the PNG's display
    gain clips to pure white, so saturation can be judged without the
    image."""
    lines = []
    for band, wavelength_nm in enumerate(main.SMARTG_WAVELENGTHS_NM):
        values = reflectance[..., band]
        q50, q99 = np.percentile(values, [50, 99])
        clipped = float(np.mean(values * gain >= 1.0))
        lines.append(
            f"{indent}{wavelength_nm:.0f} nm: min={values.min():.4f} median={q50:.4f} "
            f"p99={q99:.4f} max={values.max():.4f}  |  clipped at gain {gain:g}: {clipped:.1%}"
        )
    return "\n".join(lines)


def save_reflectance_figure(
    reflectance: np.ndarray, path: Path, title: str, gain: float
) -> None:
    """Save the raw reflectance as one heat map per wavelength band, in
    reflectance units on a shared 0-to-max colour scale (no display gain,
    gamma or clipping), with a colour bar each. Uses matplotlib's Agg
    canvas directly, so it needs no display and doesn't touch pyplot's
    global backend."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    n_bands = reflectance.shape[-1]
    vmax = float(reflectance.max())
    figure = Figure(figsize=(4.4 * n_bands, 4.6), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, n_bands, squeeze=False)[0]
    for band, axis in enumerate(axes):
        image = axis.imshow(reflectance[..., band], vmin=0.0, vmax=vmax, cmap="viridis")
        axis.set_title(f"{main.SMARTG_WAVELENGTHS_NM[band]:.0f} nm")
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(image, ax=axis, shrink=0.8, label="TOA reflectance")
    figure.suptitle(f"{title}  (raw; PNG gain {gain:g} clips at {1 / gain:.3g})")
    figure.savefig(path, dpi=110)


def run_config(config: PlaygroundConfig, log: Callable[[str], None] = print) -> None:
    """Run a saved scenario end to end, headlessly: build the geometry it
    was set up with, then do `config.renders` acquisitions (just one if the
    pointing error is off, since repeating a noise-free render only
    rewrites the same file)."""
    config.validate()
    simulation = PlaygroundSimulation.from_config(config)
    log(
        f"Scenario: {config.mode}, satellite geometry at {simulation.time_utc} "
        f"(TLE: {config.tle.splitlines()[0].strip()})"
    )
    renders = config.renders
    if renders > 1 and not config.pointing_noise_enabled:
        log("Pointing error is off, so repeated renders would be identical: doing 1.")
        renders = 1

    for k in range(1, renders + 1):
        log(f"--- render {k}/{renders} ---")
        log(simulation.render(config, on_status=log))
