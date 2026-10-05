"""SMART-G playground: an interactive sandbox for trying out radiative-
transfer techniques against the real 3D cumulus field before deciding
whether/how to wire them into main.py.

Standalone from the orbit-formation simulation: it reuses main.py's
TLE/geometry/SMART-G helpers (imported as a module, not copied), with
every viewing/illumination angle under direct interactive control
instead of being derived from real ephemeris - the point is to freely
explore geometries main.py's own pipeline doesn't (yet) expose.

Two sandbox modes (the "Sandbox mode" control)
-----------------------------------------------
"Single satellite": a lone satellite at its real current TLE position,
with a freely chosen viewing direction. The user picks it as an
(off-nadir angle, azimuth) pair relative to the satellite's OWN nadir
(straight down, toward Earth's center). Casting a ray from the satellite
in that direction and intersecting it with Earth's surface gives a
ground point, "point a". The 3D cloud field (see
main.build_smartg_scene_3d()) is centered on the axis from Earth's
center through point a - see main.viewing_angles_deg(), which
independently recovers the correct ground-based viewing (zenith,
azimuth) angles at point a (they are *not* simply the satellite's own
off-nadir angle, because of Earth's curvature).

"Formation (frozen)": the CloudCT formation exactly as main.py defines
it (main.formation_positions_km(): satellite index
main.NADIR_REFERENCE_INDEX looks at its own nadir, every other satellite
looks at that same shared ground target - see
main.render_formation_images_3d()), frozen at a single moment - no
orbital motion, unlike main.py's animated simulation. The user picks
which of the formation's satellites to render.

Both modes let the sun's direction be chosen directly - as a (zenith,
azimuth) pair relative to the target point's own local zenith/nadir -
rather than computed from real ephemeris, and let the domain's
periodic/non-periodic boundary (see build_scene_3d()) be switched at
will.

Run: python smartg_playground.py
"""

from pathlib import Path

import numpy as np
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor
from pyface.api import GUI
from skyfield.api import load
from traits.api import Bool, Button, Enum, HasTraits, Instance, Int, List, Range, Str, observe
from traitsui.api import CheckListEditor, HGroup, Item, VGroup, View

import main
import pointing_noise

OUTPUT_DIR = Path(__file__).with_name("smartg_playground_images")

# Small/fast presets for quick iteration - this is a sandbox, not the
# production renderer (see main.SMARTG_3D_RESOLUTION_PRESETS for that).
RESOLUTION_PRESETS = {
    "Tiny (16x16)": (16, 16),
    "Small (32x32)": (32, 32),
    "Medium (64x64)": (64, 64),
}
DEFAULT_RESOLUTION = "Small (32x32)"

# Kept safely inside the true horizon (~66 deg for a ~594 km CLOUDCT-
# altitude orbit, see the horizon half-angle formula in
# viewing_direction_km()'s docstring) so the chosen direction always hits
# Earth well short of grazing incidence.
MAX_OFF_NADIR_DEG = 55.0

MARKER_SCALE_KM = main.SATELLITE_MARKER_SCALE_KM
POINT_A_MARKER_SCALE_KM = MARKER_SCALE_KM * 0.6
# Length of the target -> sun arrow; the sun is effectively infinitely far
# away, so only its direction means anything, this just sets how visible it is.
SUN_ARROW_LENGTH_KM = 1500.0

DEFAULT_NOISE_PARAMS = pointing_noise.PRESETS[pointing_noise.DEFAULT_PRESET]


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


def build_scene_3d(lat_deg: float, periodic: bool):
    """Like main.build_smartg_scene_3d(), except the domain's horizontal
    boundary condition (periodic vs. non-periodic + extended margin) is a
    parameter instead of a fixed choice - this sandbox exists specifically
    to compare the two (see the finding that led to main.py fixing on
    non-periodic, in build_smartg_scene_3d()'s own docstring).

    Returns (wavelength_nm, atmosphere, surface, grid_3d), same as
    main.build_smartg_scene_3d()."""
    from smartg.albedo import AlbedoCst
    from smartg.atmosphere import Atm1D, Atm3D, Cloud3D, read_i3rc_cloud
    from smartg.grid3d import Grid3D
    from smartg.surface import LambSurface

    if not main.SMARTG_3D_CLOUD_PATH.exists():
        raise FileNotFoundError(
            f"{main.SMARTG_3D_CLOUD_PATH} not found; download it with "
            "main.ensure_smartg_auxdata(data_type='IPRT')."
        )

    field = read_i3rc_cloud(main.SMARTG_3D_CLOUD_PATH)
    cloud_3d = Cloud3D(
        main.SMARTG_CLOUD_FNAME,
        w_ref=main.SMARTG_3D_CLOUD_W_REF_NM,
        ds=field,
        reff_acc=main.SMARTG_3D_CLOUD_REFF_ACC,
        reff_min=main.SMARTG_3D_CLOUD_REFF_MIN_UM,
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
    surface = LambSurface(alb=AlbedoCst(main.SMARTG_SURFACE_ALBEDO))
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
):
    """The image's sensor grid: one sensor per pixel over the cloud field,
    all sharing one viewing direction - main.render_formation_images_3d()'s
    layout - optionally moved by a pointing error: shifted by the ground
    offset of the actual line of sight (domain x = local east, y = local
    north, the same frame as SMART-G's azimuths) and rotated about its
    centre by the yaw error's component about the local vertical.

    Shifted pixels wrap around on a periodic domain; on a non-periodic one
    they may land in the horizontal extension (cloud-free) but not beyond."""
    from smartg.grid3d import locate_voxel_index
    from smartg.sensor import Sensor, get_sensors_grid

    xgrid = np.linspace(grid_3d.xgrid[0], grid_3d.xgrid[-1], width_px + 1)
    ygrid = np.linspace(grid_3d.ygrid[0], grid_3d.ygrid[-1], height_px + 1)
    cell_size = xgrid[1] - xgrid[0]

    if offset_east_km == 0.0 and offset_north_km == 0.0 and rotation_rad == 0.0:
        return get_sensors_grid(
            xgrid,
            ygrid,
            pos_z=pos_z,
            th_deg=th_deg,
            ph_deg=ph_deg,
            loc="ATMOS",
            cell_size=cell_size,
            grid_3d=grid_3d,
        )

    # Same pixel order as get_sensors_grid(): row by row, x varying first.
    xx, yy = np.meshgrid(xgrid[:-1] + cell_size / 2.0, ygrid[:-1] + np.diff(ygrid) / 2.0)
    center_x = (xgrid[0] + xgrid[-1]) / 2.0
    center_y = (ygrid[0] + ygrid[-1]) / 2.0
    dx, dy = xx - center_x, yy - center_y
    cos_r, sin_r = np.cos(rotation_rad), np.sin(rotation_rad)
    xx = center_x + cos_r * dx - sin_r * dy + offset_east_km
    yy = center_y + sin_r * dx + cos_r * dy + offset_north_km

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
            f"pointing error moved the image footprint "
            f"({offset_east_km:+.2f} km E, {offset_north_km:+.2f} km N) past the "
            f"simulation domain's edge (+-{grid_3d.xGRID[-1]:.1f} km)"
        )

    zz = np.full(xx.size, pos_z)
    icells = locate_voxel_index(
        grid_3d.xGRID, grid_3d.yGRID, grid_3d.zGRID, xx.ravel(), yy.ravel(), zz
    )
    return [
        Sensor(
            pos_x=float(x),
            pos_y=float(y),
            pos_z=pos_z,
            th_deg=th_deg,
            ph_deg=ph_deg,
            loc="ATMOS",
            cell_size=cell_size,
            icell=int(icell),
        )
        for x, y, icell in zip(xx.ravel(), yy.ravel(), icells, strict=True)
    ]


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
    actual line of sight. The returned angles are then the actual ones."""
    main.ensure_cuda_compatible_msvc_on_path()
    from smartg.smartg import LocalEstimate, Smartg

    if n_photons is None:
        n_photons = main.SMARTG_3D_PHOTONS_PER_PIXEL * width_px * height_px
    if true_point_km is None:
        true_point_km = point_a_km

    lat_deg = main.geocentric_latitude_deg(point_a_km)
    wavelength_nm, atmosphere, surface, grid_3d = build_scene_3d(lat_deg, periodic)
    atmosphere_profile = atmosphere.calc(wavelength_nm)

    pos_z = grid_3d.zGRID[-1] - main.SMARTG_3D_SENSOR_TOP_OFFSET_KM

    vza_deg, vaa_deg = main.viewing_angles_deg(satellite_position_km, true_point_km)
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

    lat_deg = main.geocentric_latitude_deg(target_km)
    wavelength_nm, atmosphere, surface, grid_3d = build_scene_3d(lat_deg, periodic)
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
        vza_deg, vaa_deg = main.viewing_angles_deg(positions_km[index], true_point_km)
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


class SmartgPlayground(HasTraits):
    """3D scene plus controls for two sandbox modes (see the module
    docstring), the sun direction, the domain's periodic/non-periodic
    boundary, and a Render button that runs SMART-G and saves the
    resulting image(s).

    "Single satellite": the satellite, and the chosen viewing ray down to
    point a. "Formation (frozen)": the CloudCT formation's 10 satellites,
    their pointing arrows, and the shared nadir target they all point at.
    """

    scene = Instance(MlabSceneModel, ())

    mode = Enum("Single satellite", "Formation (frozen)")

    off_nadir_deg = Range(0.0, MAX_OFF_NADIR_DEG, 0.0)
    view_azimuth_deg = Range(0.0, 360.0, 0.0)
    # Formation mode: which satellites to render - all 10 checked would
    # mean 10 sequential Smartg.run() calls, so this defaults to just the
    # nadir reference satellite for a fast first render.
    satellites_to_render = List(Str, [f"Sat {main.NADIR_REFERENCE_INDEX}"])

    sza_deg = Range(0.0, 100.0, 30.0)
    saa_deg = Range(0.0, 360.0, 90.0)
    periodic = Bool(False)
    resolution = Enum(list(RESOLUTION_PRESETS))
    # Default matches what render_view()/render_formation_frozen() assume
    # when n_photons isn't given; the total sent to SMART-G is this times
    # the pixel count.
    photons_per_pixel = Range(
        1.0, 1.0e6, main.SMARTG_3D_PHOTONS_PER_PIXEL, mode="text"
    )

    # Pointing error (see pointing_noise.py). Sigmas are 1 sigma per
    # cross-boresight axis; picking a preset fills them in, editing any of
    # them switches the preset to "Custom".
    pointing_noise_enabled = Bool(False)
    noise_preset = Enum(
        pointing_noise.DEFAULT_PRESET, [*pointing_noise.PRESETS, "Custom"]
    )
    sigma_bias_deg = Range(0.0, 10.0, DEFAULT_NOISE_PARAMS["sigma_bias_deg"], mode="text")
    sigma_acquisition_deg = Range(
        0.0, 10.0, DEFAULT_NOISE_PARAMS["sigma_acquisition_deg"], mode="text"
    )
    sigma_drift_deg = Range(0.0, 10.0, DEFAULT_NOISE_PARAMS["sigma_drift_deg"], mode="text")
    drift_tau_s = Range(1.0, 1.0e7, DEFAULT_NOISE_PARAMS["drift_tau_s"], mode="text")
    sigma_jitter_deg = Range(0.0, 10.0, DEFAULT_NOISE_PARAMS["sigma_jitter_deg"], mode="text")
    yaw_factor = Range(0.0, 10.0, pointing_noise.DEFAULT_YAW_FACTOR, mode="text")
    acquisition_interval_s = Range(
        0.0, 1.0e7, pointing_noise.DEFAULT_ACQUISITION_INTERVAL_S, mode="text"
    )
    noise_seed = Int(0)
    new_run_button = Button("New Monte-Carlo run")
    noise_budget_text = Str()

    render_button = Button("Render")
    status_text = Str()

    view = View(
        VGroup(
            Item(
                "scene",
                editor=SceneEditor(scene_class=MayaviScene),
                height=600,
                width=900,
                show_label=False,
            ),
            Item("mode", label="Sandbox mode"),
            HGroup(
                Item("off_nadir_deg", label="Off-nadir angle (deg)"),
                Item("view_azimuth_deg", label="View azimuth (deg)"),
                visible_when="mode == 'Single satellite'",
            ),
            Item(
                "satellites_to_render",
                label="Satellites to render",
                style="custom",
                editor=CheckListEditor(
                    values=[f"Sat {i}" for i in range(main.NUM_SATELLITES)],
                    cols=main.NUM_SATELLITES,
                ),
                visible_when="mode == 'Formation (frozen)'",
            ),
            HGroup(
                Item("sza_deg", label="Sun zenith angle (deg)"),
                Item("saa_deg", label="Sun azimuth (deg)"),
            ),
            HGroup(
                Item("periodic", label="Periodic domain"),
                Item("resolution", label="Image resolution"),
                Item("photons_per_pixel", label="Photons per pixel"),
                Item("render_button", show_label=False),
            ),
            VGroup(
                HGroup(
                    Item("pointing_noise_enabled", label="Enable"),
                    Item("noise_preset", label="Preset"),
                    Item("noise_seed", label="Random seed"),
                    Item("new_run_button", show_label=False),
                ),
                HGroup(
                    Item("sigma_bias_deg", label="Static bias sigma (deg)"),
                    Item("sigma_acquisition_deg", label="Per-acquisition sigma (deg)"),
                    Item("sigma_drift_deg", label="Drift sigma (deg)"),
                    Item("drift_tau_s", label="Drift time constant (s)"),
                    enabled_when="pointing_noise_enabled",
                ),
                HGroup(
                    Item("sigma_jitter_deg", label="Jitter sigma (deg)"),
                    Item("yaw_factor", label="Yaw sigma factor"),
                    Item("acquisition_interval_s", label="Time between renders (s)"),
                    enabled_when="pointing_noise_enabled",
                ),
                Item("noise_budget_text", show_label=False, style="readonly"),
                label="Pointing error",
                show_border=True,
            ),
            Item("status_text", show_label=False, style="readonly"),
        ),
        resizable=True,
        title="SMART-G Playground",
    )

    def __init__(self, satellite, t, **traits) -> None:
        super().__init__(**traits)
        self.satellite = satellite
        self.t = t
        self.position_km = satellite.at(t).frame_xyz(main.itrs).km

        # The formation's positions, frozen at this same single moment t -
        # see main.formation_positions_km()/main.render_formation_images_3d().
        self._formation_time_offsets_seconds = main.formation_time_offsets_seconds(
            satellite, t
        )
        self._formation_positions_km = main.formation_positions_km(
            satellite, t, self._formation_time_offsets_seconds
        )
        self._formation_target_km = main.nadir_target_km(
            self._formation_positions_km[main.NADIR_REFERENCE_INDEX]
        )

        self._satellite_marker = None
        self._direction_line = None
        self._point_a_marker = None
        self.point_a_km = None
        self._recompute_point_a()

        self._formation_points = None
        self._formation_arrows = None
        self._formation_target_marker = None
        self._sun_arrow = None

        # Along-track direction for each satellite's camera frame (roll
        # axis): the orbit normal crossed with its position. Every
        # satellite shares the orbital plane, so one normal serves all.
        self._orbit_normal = main.orbital_normal_itrs(satellite, t)
        self._noise = pointing_noise.PointingNoise(self.noise_seed)
        self._applying_preset = False
        self._update_noise_budget()

    @observe("scene.activated")
    def _setup_scene(self, event=None) -> None:
        self.scene.background = (0.0, 0.0, 0.0)
        texture_path = main.ensure_earth_texture()
        main.add_earth(self.scene, texture_path)

        self._satellite_marker = self.scene.mlab.points3d(
            [self.position_km[0]],
            [self.position_km[1]],
            [self.position_km[2]],
            color=(1.0, 0.1, 0.1),
            scale_factor=MARKER_SCALE_KM,
        )
        self._redraw_direction()

        self._setup_formation_actors()
        self._apply_mode_visibility()
        self._center_view_on_target()

    def _setup_formation_actors(self) -> None:
        positions = self._formation_positions_km
        pointing_vectors = main.formation_pointing_vectors_km(positions)
        self._formation_points = self.scene.mlab.points3d(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            color=(1.0, 0.1, 0.1),
            scale_factor=MARKER_SCALE_KM,
        )
        self._formation_arrows = self.scene.mlab.quiver3d(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            pointing_vectors[:, 0],
            pointing_vectors[:, 1],
            pointing_vectors[:, 2],
            color=(0.2, 1.0, 1.0),
            mode="arrow",
            scale_factor=1.0,
            scale_mode="vector",
        )
        target = self._formation_target_km
        self._formation_target_marker = self.scene.mlab.points3d(
            [target[0]],
            [target[1]],
            [target[2]],
            color=(1.0, 0.85, 0.0),
            scale_factor=POINT_A_MARKER_SCALE_KM,
        )

    def _apply_mode_visibility(self) -> None:
        if self._satellite_marker is None or self._formation_points is None:
            return  # scene not yet activated
        single = self.mode == "Single satellite"
        self._satellite_marker.visible = single
        self._direction_line.visible = single
        self._point_a_marker.visible = single
        self._formation_points.visible = not single
        self._formation_arrows.visible = not single
        self._formation_target_marker.visible = not single

    def _current_target_km(self) -> np.ndarray:
        """The spot on Earth's surface the satellite(s) look at."""
        if self.mode == "Single satellite":
            return self.point_a_km
        return self._formation_target_km

    def _center_view_on_target(self) -> None:
        """Make the camera orbit/zoom about the spot the satellite(s) look
        at (point a, or the formation's shared nadir target) instead of the
        Earth's centre. Only the focal point moves; the viewing direction
        and distance are kept. Also keeps the sun arrow anchored there."""
        if self._satellite_marker is None:
            return  # scene not yet activated
        # Arrow first: creating a new actor can make Mayavi re-fit the
        # camera to the whole scene (i.e. back onto Earth's centre), which
        # would undo the focal point set below.
        self._redraw_sun_arrow()
        target = self._current_target_km()
        self.scene.mlab.view(focalpoint=[float(c) for c in target])

    def _redraw_sun_arrow(self) -> None:
        """Arrow from the target spot toward the sun, per the GUI's SZA/SAA
        (zenith from local up, azimuth counterclockwise from local east -
        same convention as viewing_direction_km() and main.enu_basis_km())."""
        if self._satellite_marker is None:
            return  # scene not yet activated
        target = self._current_target_km()
        east, north, up = main.enu_basis_km(target)
        zenith = np.radians(self.sza_deg)
        azimuth = np.radians(self.saa_deg)
        horizontal = east * np.cos(azimuth) + north * np.sin(azimuth)
        direction = up * np.cos(zenith) + horizontal * np.sin(zenith)
        direction = direction / np.linalg.norm(direction)
        x, y, z = ([float(c)] for c in target)
        u, v, w = ([float(c)] for c in direction)
        if self._sun_arrow is None:
            self._sun_arrow = self.scene.mlab.quiver3d(
                x,
                y,
                z,
                u,
                v,
                w,
                color=(1.0, 0.5, 0.0),
                mode="arrow",
                scale_factor=SUN_ARROW_LENGTH_KM,
                scale_mode="vector",
            )
        else:
            self._sun_arrow.mlab_source.reset(x=x, y=y, z=z, u=u, v=v, w=w)

    def _sza_deg_changed(self) -> None:
        self._redraw_sun_arrow()

    def _saa_deg_changed(self) -> None:
        self._redraw_sun_arrow()

    def _mode_changed(self) -> None:
        self._apply_mode_visibility()
        self._center_view_on_target()
        self.status_text = f"Switched to {self.mode} mode."

    def _recompute_point_a(self) -> None:
        direction = viewing_direction_km(
            self.position_km, self.off_nadir_deg, self.view_azimuth_deg
        )
        self.point_a_km = main.ray_sphere_intersection_km(
            self.position_km, direction, main.EARTH_RADIUS_KM
        )

    def _redraw_direction(self) -> None:
        if self._satellite_marker is None:
            return  # scene not yet activated
        self._recompute_point_a()
        xs = [self.position_km[0], self.point_a_km[0]]
        ys = [self.position_km[1], self.point_a_km[1]]
        zs = [self.position_km[2], self.point_a_km[2]]
        if self._direction_line is None:
            self._direction_line = self.scene.mlab.plot3d(
                xs, ys, zs, color=(0.2, 1.0, 0.2), tube_radius=None, line_width=2.0
            )
            self._point_a_marker = self.scene.mlab.points3d(
                [self.point_a_km[0]],
                [self.point_a_km[1]],
                [self.point_a_km[2]],
                color=(1.0, 0.85, 0.0),
                scale_factor=POINT_A_MARKER_SCALE_KM,
            )
        else:
            self._direction_line.mlab_source.reset(x=xs, y=ys, z=zs)
            self._point_a_marker.mlab_source.reset(
                x=[self.point_a_km[0]], y=[self.point_a_km[1]], z=[self.point_a_km[2]]
            )

    def _off_nadir_deg_changed(self) -> None:
        self._redraw_direction()
        self._center_view_on_target()

    def _view_azimuth_deg_changed(self) -> None:
        self._redraw_direction()
        self._center_view_on_target()

    def _noise_params(self) -> pointing_noise.PointingNoiseParams:
        return pointing_noise.PointingNoiseParams(
            sigma_bias_deg=self.sigma_bias_deg,
            sigma_acquisition_deg=self.sigma_acquisition_deg,
            sigma_drift_deg=self.sigma_drift_deg,
            drift_tau_s=self.drift_tau_s,
            sigma_jitter_deg=self.sigma_jitter_deg,
            yaw_factor=self.yaw_factor,
            acquisition_interval_s=self.acquisition_interval_s,
        )

    def _update_noise_budget(self) -> None:
        """Budget check (the document's step 7): APE 1 sigma per axis by
        RSS, its cone CE90 by Rayleigh statistics, and roughly what that
        means on the ground at nadir."""
        sigma_deg = self._noise_params().ape_sigma_deg()
        ce90_deg = pointing_noise.CE90_PER_SIGMA * sigma_deg
        altitude_km = np.linalg.norm(self.position_km) - main.EARTH_RADIUS_KM
        ce90_ground_km = altitude_km * np.radians(ce90_deg)
        self.noise_budget_text = (
            f"APE 1-sigma per axis = {sigma_deg:.4f} deg  |  "
            f"cone CE90 = {ce90_deg:.4f} deg = ~{ce90_ground_km:.2f} km on the ground at nadir  |  "
            f"Monte-Carlo run seed {self.noise_seed}, renders so far: {self._noise.acquisition}"
        )

    @observe(
        [
            "sigma_bias_deg",
            "sigma_acquisition_deg",
            "sigma_drift_deg",
            "drift_tau_s",
            "sigma_jitter_deg",
        ]
    )
    def _noise_sigma_changed(self, event) -> None:
        if not self._applying_preset:
            values = {
                name: getattr(self, name)
                for name in pointing_noise.PRESETS[pointing_noise.DEFAULT_PRESET]
            }
            self.noise_preset = next(
                (name for name, preset in pointing_noise.PRESETS.items() if preset == values),
                "Custom",
            )
        self._update_noise_budget()

    @observe(["yaw_factor", "acquisition_interval_s"])
    def _noise_other_changed(self, event) -> None:
        self._update_noise_budget()

    def _noise_preset_changed(self) -> None:
        if self.noise_preset == "Custom":
            return
        self._applying_preset = True
        try:
            for name, value in pointing_noise.PRESETS[self.noise_preset].items():
                setattr(self, name, value)
        finally:
            self._applying_preset = False
        self._update_noise_budget()

    def _noise_seed_changed(self) -> None:
        self._noise.reset(self.noise_seed)
        self._update_noise_budget()

    def _new_run_button_fired(self) -> None:
        # A new seed = new static biases and drift states for every satellite.
        self.noise_seed += 1

    def _draw_pointing(self, key, position_km: np.ndarray, commanded_point_km: np.ndarray):
        """Draw this acquisition's pointing error for one satellite and
        ray-trace it: returns (true ground point km, footprint rotation rad,
        short summary text)."""
        boresight = commanded_point_km - position_km
        boresight /= np.linalg.norm(boresight)
        along_track = np.cross(self._orbit_normal, position_km)
        error_rad = self._noise.draw_error_rad(key, self._noise_params())
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

    def _noise_file_suffix(self) -> str:
        if not self.pointing_noise_enabled:
            return ""
        return f"_noise_seed{self.noise_seed}_render{self._noise.acquisition}"

    def _render_button_fired(self) -> None:
        width_px, height_px = RESOLUTION_PRESETS[self.resolution]
        if self.pointing_noise_enabled:
            # Every Render click is a new acquisition (new target / slew).
            self._noise.begin_acquisition()
            self._update_noise_budget()
        if self.mode == "Single satellite":
            self._render_single(width_px, height_px)
        else:
            self._render_formation(width_px, height_px)

    def _render_single(self, width_px: int, height_px: int) -> None:
        self._recompute_point_a()
        self.status_text = (
            f"Rendering at {width_px}x{height_px} "
            f"(periodic={self.periodic}, off-nadir={self.off_nadir_deg:.1f} deg, "
            f"view az={self.view_azimuth_deg:.1f} deg)..."
        )
        GUI.process_events()

        true_point_km, rotation_rad, pointing_summary = self.point_a_km, 0.0, ""
        if self.pointing_noise_enabled:
            true_point_km, rotation_rad, pointing_summary = self._draw_pointing(
                "single", self.position_km, self.point_a_km
            )
            pointing_summary += "  |  "

        try:
            reflectance, vza_deg, vaa_deg = render_view(
                self.position_km,
                self.point_a_km,
                self.sza_deg,
                self.saa_deg,
                self.periodic,
                width_px,
                height_px,
                n_photons=self.photons_per_pixel * width_px * height_px,
                true_point_km=true_point_km,
                footprint_rotation_rad=rotation_rad,
            )
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            path = OUTPUT_DIR / (
                f"single_offnadir{self.off_nadir_deg:.0f}_az{self.view_azimuth_deg:.0f}_"
                f"sza{self.sza_deg:.0f}_saa{self.saa_deg:.0f}_"
                f"{'periodic' if self.periodic else 'nonperiodic'}"
                f"{self._noise_file_suffix()}.png"
            )
            # Periodic domains run much brighter (see
            # build_smartg_scene_3d()'s docstring) - a rough compensation,
            # not exact for every geometry.
            gain = 3.0 if self.periodic else main.SMARTG_3D_DEFAULT_GAIN
            main.reflectance_to_image(reflectance, gain=gain, gamma=1.8).save(path)
        except Exception as exc:  # a GPU/auxdata/SMART-G failure shouldn't crash the GUI
            self.status_text = f"Render failed: {exc}"
        else:
            self.status_text = (
                f"{pointing_summary}"
                f"Ground VZA={vza_deg:.1f} VAA={vaa_deg:.1f} deg at point a  |  "
                f"reflectance mean={reflectance.mean():.5f} max={reflectance.max():.5f}  |  "
                f"saved {path}"
            )

    def _render_formation(self, width_px: int, height_px: int) -> None:
        if not self.satellites_to_render:
            self.status_text = "Select at least one satellite first."
            return

        satellite_indices = sorted(
            int(label.split()[1]) for label in self.satellites_to_render
        )
        self.status_text = (
            f"Rendering {len(satellite_indices)} satellite(s) at {width_px}x{height_px} "
            f"(periodic={self.periodic})..."
        )
        GUI.process_events()

        pointing = {}
        summaries = []
        if self.pointing_noise_enabled:
            for index in satellite_indices:
                true_point_km, rotation_rad, summary = self._draw_pointing(
                    index, self._formation_positions_km[index], self._formation_target_km
                )
                pointing[index] = (true_point_km, rotation_rad)
                summaries.append(f"Sat {index}: {summary}")

        try:
            images = render_formation_frozen(
                self._formation_target_km,
                self._formation_positions_km,
                satellite_indices,
                self.sza_deg,
                self.saa_deg,
                self.periodic,
                width_px,
                height_px,
                n_photons=self.photons_per_pixel * width_px * height_px,
                pointing=pointing,
            )
            subdir = OUTPUT_DIR / (
                f"formation_sza{self.sza_deg:.0f}_saa{self.saa_deg:.0f}_"
                f"{'periodic' if self.periodic else 'nonperiodic'}"
                f"{self._noise_file_suffix()}"
            )
            subdir.mkdir(parents=True, exist_ok=True)
            gain = 3.0 if self.periodic else main.SMARTG_3D_DEFAULT_GAIN
            paths = []
            for index in sorted(images):
                path = subdir / f"satellite_{index:02d}.png"
                main.reflectance_to_image(images[index], gain=gain, gamma=1.8).save(path)
                paths.append(path)
        except Exception as exc:  # a GPU/auxdata/SMART-G failure shouldn't crash the GUI
            self.status_text = f"Render failed: {exc}"
        else:
            self.status_text = "\n".join(
                [f"Saved {len(paths)} image(s) to {subdir}", *summaries]
            )


def run_playground() -> None:
    """Entry point. Named to avoid colliding with the `main` module this
    file imports everything from."""
    tle_text = main.get_tle()
    satellite = main.build_satellite(tle_text)
    ts = load.timescale()
    t = ts.now()
    app = SmartgPlayground(satellite, t)
    app.configure_traits()


if __name__ == "__main__":
    run_playground()
