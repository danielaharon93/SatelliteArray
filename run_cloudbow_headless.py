"""Intensity and degree of linear polarization images against scattering angle, for BOMEX clouds - headlessly.

The cloudbow (the cloud's version of a rainbow) sits at scattering angles of about
135-150 degrees, and shows most clearly in the polarization. This script renders the
scenario of a saved config (the satellite, its view, the surface, the atmosphere, the
resolution, the photons...) on each BOMEX cloud once per scattering angle, at ONE wavelength
(WAVELENGTH_NM, 645 nm by default: the "single" band set), and makes two sets of 2D heat maps,
one image per scattering angle in each:

    intensity/   the intensity I of each pixel (reflectance units)
    dolp/        the degree of linear polarization of each pixel, sqrt(Q^2 + U^2) / I, in per cent

with the same colour scale at every angle, so the angles can be compared, and the same images
of all the angles in one PDF per set (intensity_all_angles.pdf, dolp_all_angles.pdf). A pixel's
own DoLP is noisy, and biased high where the pixel is dark or has few photons (Monte Carlo
noise in Q and U does not average out in sqrt(Q^2 + U^2)): raise PHOTONS_PER_PIXEL to see the
bow in the images. The Stokes parameters of every angle are also saved by the render (*_stokes.npy).

The sun is fixed by the scenario (its zenith and azimuth), so the scattering angle is changed
by moving the CAMERA, in the vertical plane of the sun, at the signed view zenith t that gives
each scattering angle:

    scattering angle = 180 deg - |sun zenith - t|    (t > 0: the sun's side, t < 0: the other side)

The camera passes through the nadir at a scattering angle of 180 deg - sun zenith. For each
direction the script solves the satellite's pointing (off-nadir angle and azimuth) that gives
it (the scenario's own pointing is replaced), and prints it. The general formula, which the
script also prints for each angle as a check, is
cos(scattering angle) = -(cos(sza) cos(vza) + sin(sza) sin(vza) cos(saa - vaa)).
Camera directions the Earth shadows (the satellite cannot see the cloud from there), or that
need an off-nadir angle above the playground's limit, are skipped with a warning.

The camera's direction also changes the view of the cloud, so the images of different angles
look at the cloud from different directions.

A non-periodic domain is a box with an empty margin around the cloud field (main.py's 30 km),
which must hold the sideways shift of the render's sensors (for an oblique camera) and the
sideways reach of the sun's rays. HORIZ_MARGIN_KM sets it (None: automatic, from the camera
and the sun), as in run_cloudbow_voxel_headless.py; main.py's value is put back after the run.
A periodic domain has no margin.

The cloud's physics - how the extinction is computed (extinction_method), the crop
(bomex_crop_cells) and the droplet phase function (phase_function, hg_g) - is the
scenario's own, from the config file: set it there (in the GUI, "Save configuration").
This script only chooses which BOMEX files to run and where things go (SETTINGS below).
The pointing error is switched off here (the pointing is solved exactly for each angle).

Needs a CUDA GPU and SMART-G's auxiliary data, as run_bomex_headless.py does, unless
PLAN_ONLY is True. Run:

    python run_cloudbow_headless.py
"""

import csv
import dataclasses
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# ============================== SETTINGS ==============================
# Where the BOMEX netCDF files are: the folder, and which files in it (the pattern is searched
# in the folder only; use "**/*.com3D_int_2.nc" to include its subfolders). Files without the
# FNCD droplet spectrum are skipped with a message. A relative path is taken relative to this
# file's folder; an absolute path also works.
BOMEX_DIR = "/data/bomex"
BOMEX_PATTERN = "*.com3D_int_2.nc"
# Or name the files yourself (full paths); when not empty, BOMEX_DIR and BOMEX_PATTERN are ignored.
BOMEX_FILES: list[str] = []

# The saved scenario to render on each cloud (saved from the GUI with "Save configuration"):
# the satellite and its view, the surface, the atmosphere, the band set, the resolution, the
# photons, and the cloud's physics (extinction method, crop, phase function). Its cloud source and
# BOMEX file settings are replaced by the files above, and so is its water body setting, which a
# cloud scene does not support.
SCENARIO_CONFIG = "configs/scenario.json"

# Where the generated cloud files go (one per dataset and setting, reused on later runs).
GENERATED_CLOUD_DIR = "generated_clouds"

# The scattering angles, in degrees: from START to STOP (included) every STEP.
ANGLE_START_DEG = 135.0
ANGLE_STOP_DEG = 150.0
ANGLE_STEP_DEG = 2.5

# None keeps the scenario's value. Each angle is a whole render, so a smaller image or fewer
# photons per pixel is the way to a faster curve. RESOLUTION is a name from
# playground_core.RESOLUTION_PRESETS, e.g. "Small (32x32)".
RESOLUTION: str | None = None
PHOTONS_PER_PIXEL: float | None = None
# The wavelengths also set the cost: SMART-G builds a phase matrix per wavelength and per droplet
# radius, for every angle. A coarser GRID_STEP_NM (for "grid" and "grid_modis") is much cheaper.
# One wavelength: the "single" band set at WAVELENGTH_NM (645 nm). SMART-G runs with several wavelengths
# lost the cloud's Mie phase function (and so the bow) in the tests made for run_cloudbow_voxel_headless.py,
# so this script uses one. None keeps the scenario's value; a band set with several wavelengths prints a warning.
BAND_SET: str | None = "single"
WAVELENGTH_NM: float | None = 645.0
CHANNEL: str | None = None
GRID_STEP_NM: float | None = None

# The empty margin of the simulation box around a non-periodic cloud field, in km. None: automatic,
# from the camera's largest view zenith and the sun's zenith (never below main.py's 30 km). A number
# sets it; camera directions that do not fit in it are skipped with a warning. A periodic domain has
# no margin and ignores this.
HORIZ_MARGIN_KM: float | None = None

# True: only print the camera directions and scattering angles of the plan (no render; needs
# no GPU and no SMART-G).
PLAN_ONLY = False

# The colour scales of the images, the same at every angle: the top of the scale is this value (reflectance
# for the intensity, per cent for the DoLP), or, when None, the 99.5th percentile of all the pixels of all the
# angles (so a few very bright or noisy pixels do not set it). The bottom is 0.
INTENSITY_MAX: float | None = None
DOLP_MAX_PERCENT: float | None = None

# One folder per dataset, in it one folder per angle (the render's files, as in run_bomex_headless.py),
# the folders intensity and dolp (one PNG per angle), the PDFs of all the angles and cloudbow_angles.csv.
OUTPUT_DIR = "cloudbow_runs"
# =======================================================================

here = Path(__file__).parent


def resolve(path: str) -> Path:
    return (here / path).expanduser()  # an absolute path wins


# The no-display setup (Traits backend, SMARTG_DIR_AUXDATA, the import workaround for main.py)
# lives in run_from_config.py; importing it only does that setup and does not run anything.
import run_from_config  # noqa: E402

playground_core = run_from_config.playground_core
main = playground_core.main
DEFAULT_MARGIN_KM = main.SMARTG_3D_HORIZ_EXTEND_KM  # main.py's margin, kept to restore after a run


def find_datasets() -> list[Path]:
    if BOMEX_FILES:
        return [resolve(f) for f in BOMEX_FILES]
    folder = resolve(BOMEX_DIR)
    if not folder.is_dir():
        sys.exit(f"BOMEX_DIR {folder} is not a folder: set BOMEX_DIR at the top of {Path(__file__).name}")
    return sorted(folder.glob(BOMEX_PATTERN))


def has_droplet_spectrum(path: Path) -> bool:
    from netCDF4 import Dataset

    with Dataset(path) as ds:
        return "FNCD" in ds.variables


def scattering_angle_deg(sza_deg: float, saa_deg: float, vza_deg: float, vaa_deg: float) -> float:
    """The angle between the light's direction before the scattering (from the sun, down) and after
    it (toward the camera): 180 deg is the exact backscatter. All four angles are in SMART-G's
    local frame (zenith from straight up, azimuth counterclockwise from east, of the direction
    toward the sun and toward the camera)."""
    sza, vza, dphi = np.radians(sza_deg), np.radians(vza_deg), np.radians(saa_deg - vaa_deg)
    cos_angle = np.cos(sza) * np.cos(vza) + np.sin(sza) * np.sin(vza) * np.cos(dphi)
    return float(np.degrees(np.arccos(np.clip(-cos_angle, -1.0, 1.0))))


def camera_for_scattering_angle(theta_deg: float, sza_deg: float, saa_deg: float) -> tuple[float, float]:
    """(view zenith, view azimuth) of the camera that gives the scattering angle for the fixed sun (sza,
    saa): in the vertical plane of the sun, at the signed zenith t = sza - (180 - theta) - on the sun's side
    for t >= 0, on the opposite side (azimuth + 180) for t < 0 - the way through the nadir that keeps the
    view zenith smallest. The scattering angle is then 180 - |sza - t|."""
    t = sza_deg - (180.0 - theta_deg)
    if t >= 0.0:
        return t, saa_deg % 360.0
    return -t, (saa_deg + 180.0) % 360.0


def view_from_pointing(simulation, off_nadir_deg: float, azimuth_deg: float, height_km: float) -> tuple[float, float]:
    """The (view zenith, view azimuth) at the cloud, at height_km, when the scenario's satellite points
    off_nadir_deg away from nadir toward azimuth_deg: what the render uses as the camera's direction."""
    point_a_km = simulation.point_a_km(off_nadir_deg, azimuth_deg)
    return main.viewing_angles_deg(simulation.position_km, main.aim_point_km(point_a_km, height_km))


def pointing_for_view(simulation, height_km: float, vza_deg: float, vaa_deg: float):
    """(off-nadir, pointing azimuth), in degrees, with which the satellite looks at the cloud from the view
    direction (vza, vaa), or None when no pointing does (beyond the Earth's limb, or no convergence). The
    law of sines gives the start (sin(off-nadir) = Earth radius x sin(view zenith) / orbit radius; the
    azimuth is opposite), and a Newton iteration with a numerical Jacobian finishes it."""
    radius, orbit = main.EARTH_RADIUS_KM, float(np.linalg.norm(simulation.position_km))
    limb = float(np.degrees(np.arcsin(radius / orbit)))  # the off-nadir of the ray that grazes the Earth
    if vza_deg < 1e-3:  # straight above the cloud: the nadir, whatever the azimuth
        return 0.0, 0.0
    wrap = lambda angle: (angle + 180.0) % 360.0 - 180.0
    eta = float(np.degrees(np.arcsin(min(1.0, radius * np.sin(np.radians(vza_deg)) / orbit))))
    phi = (vaa_deg + 180.0) % 360.0
    step = 1e-4

    def residual(e, f):
        vz, va = view_from_pointing(simulation, e, f, height_km)
        return np.array([vz - vza_deg, wrap(va - vaa_deg)])

    for _ in range(40):
        r = residual(eta, phi)
        if np.abs(r).max() < 1e-8:
            return (eta, phi % 360.0) if eta < limb else None
        jacobian = np.column_stack([(residual(eta + step, phi) - r) / step, (residual(eta, phi + step) - r) / step])
        try:
            delta = np.linalg.solve(jacobian, -r)
        except np.linalg.LinAlgError:
            return None
        eta = float(np.clip(eta + np.clip(delta[0], -5.0, 5.0), 0.0, limb - 1e-6))
        phi = (phi + float(np.clip(delta[1], -30.0, 30.0))) % 360.0
    return None


def grid_top_km(dat_path: Path) -> float:
    """The height of the top of the cloud file's grid (km): where the render's sensors sit."""
    from smartg.atmosphere import read_i3rc_cloud

    return float(read_i3rc_cloud(dat_path)["z_bounds"].values[-1])


def automatic_margin_km(top_km: float, height_km: float, sza_deg: float, max_vza_deg: float) -> float:
    """The margin of the box (km) that holds both the sideways shift of the render's sensors for the largest
    view zenith and the sideways reach of the sun's rays for the sun zenith: (top - cloud height) x tan(the
    larger angle), times 1.25, rounded up to a multiple of 10 km, and at least main.py's own margin."""
    shift = (top_km - height_km) * np.tan(np.radians(max(sza_deg, max_vza_deg)))
    return float(max(DEFAULT_MARGIN_KM, 10.0 * np.ceil(1.25 * shift / 10.0)))


def camera_limit_deg(margin_km: float, top_km: float, height_km: float) -> float:
    """The largest view zenith (deg) the render can use in a non-periodic box with this margin: the shift of its
    sensors, (top - cloud height) x tan(view zenith), must be less than the margin."""
    return float(np.degrees(np.arctan(margin_km / (top_km - height_km))))


def angles() -> list[float]:
    if ANGLE_STEP_DEG <= 0 or ANGLE_STOP_DEG < ANGLE_START_DEG or ANGLE_STOP_DEG > 180.0:
        sys.exit("Set ANGLE_START_DEG <= ANGLE_STOP_DEG <= 180 and ANGLE_STEP_DEG > 0.")
    n = int(np.floor((ANGLE_STOP_DEG - ANGLE_START_DEG) / ANGLE_STEP_DEG + 1e-9)) + 1
    return [float(ANGLE_START_DEG + k * ANGLE_STEP_DEG) for k in range(n)]


def direct_sun_limit_deg(dat_path: Path, cloud_height_km: float) -> float:
    """The sun zenith angle above which, in a NON-periodic domain, the direct sun does not reach the cloud:
    the sun ray through the cloud crosses the top of the grid (about 30 km up) at a horizontal distance of
    (top - cloud height) x tan(sun zenith) from it, and the grid reaches only its horizontal extension
    (main.SMARTG_3D_HORIZ_EXTEND_KM) and half the field beyond the cloud's centre. A periodic domain has no
    such limit."""
    from smartg.atmosphere import read_i3rc_cloud

    field = read_i3rc_cloud(dat_path)
    z_top = float(field["z_bounds"].values[-1])
    half_width = 0.5 * float(field["x_bounds"].values[-1] - field["x_bounds"].values[0])
    reach = main.SMARTG_3D_HORIZ_EXTEND_KM + half_width
    return float(np.degrees(np.arctan(reach / max(z_top - cloud_height_km, 1e-6))))


def stokes_maps(stokes: np.ndarray) -> dict:
    """(4, h, w, n_channels) Stokes parameters of a one-wavelength render -> the (h, w) intensity I and the (h, w)
    DoLP of each pixel (0 to 1), and the image's mean I and the DoLP of its mean Stokes parameters."""
    if stokes.shape[-1] != 1:
        raise ValueError(f"the render has {stokes.shape[-1]} channels: this script needs one wavelength (BAND_SET \"single\")")
    images = stokes[..., 0]  # (4, h, w)
    dolp, _ = playground_core.stokes_to_polarization(images)
    mean = images.reshape(4, -1).mean(axis=1)
    mean_dolp, _ = playground_core.stokes_to_polarization(mean[:, None])
    return {"intensity": images[0], "dolp": np.asarray(dolp), "mean_intensity": float(mean[0]), "mean_dolp": float(mean_dolp[0])}


def write_csv(folder: Path, rows: list[dict], sun: tuple) -> None:
    """cloudbow_angles.csv: one line per angle, with the camera's direction and pointing, the image's mean
    intensity, and the DoLP of the image's mean Stokes parameters (a whole-image number, the sea included)."""
    sza, saa = sun
    with (folder / "cloudbow_angles.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scattering_angle_deg", "view_zenith_deg", "view_azimuth_deg", "off_nadir_deg",
                         "pointing_azimuth_deg", "sun_zenith_deg", "sun_azimuth_deg", "image_mean_intensity",
                         "image_mean_stokes_dolp_percent"])
        for row in rows:
            writer.writerow([f"{row['theta']:.3f}", f"{row['vza']:.3f}", f"{row['vaa']:.3f}", f"{row['eta']:.3f}",
                             f"{row['phi']:.3f}", f"{sza:.3f}", f"{saa:.3f}", f"{row['mean_intensity']:.6g}",
                             f"{100 * row['mean_dolp']:.4f}"])


def write_heatmaps(folder: Path, dataset: str, label: str, rows: list[dict], sun: tuple, config) -> None:
    """The two sets of heat maps: intensity/ and dolp/ (one PNG per angle, the same colour scale at every angle)
    and intensity_all_angles.pdf and dolp_all_angles.pdf (all the angles in one figure)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sza, saa = sun
    sets = (
        ("intensity", "intensity", "Intensity I (reflectance)", 1.0, INTENSITY_MAX),
        ("dolp", "dolp", "DoLP (%)", 100.0, DOLP_MAX_PERCENT),
    )
    for name, key, quantity, scale, fixed_max in sets:
        values = [scale * row[key] for row in rows]
        vmax = fixed_max if fixed_max is not None else float(np.percentile(np.concatenate([v.ravel() for v in values]), 99.5))
        vmax = vmax if vmax > 0 else 1.0
        (folder / name).mkdir(exist_ok=True)
        subtitle = (f"{dataset}, {label}, {config.resolution}, {config.photons_per_pixel:g} photons/pixel, "
                    f"sun zenith {sza:.1f} deg, azimuth {saa:.1f} deg")
        for row, image in zip(rows, values):
            figure, axis = plt.subplots(figsize=(5.8, 5.0))
            shown = axis.imshow(image, vmin=0.0, vmax=vmax, cmap="viridis")
            figure.colorbar(shown, ax=axis, label=quantity, fraction=0.046, pad=0.04)
            axis.set_title(f"{quantity}: scattering angle {row['theta']:g} deg\nview zenith {row['vza']:.1f} deg, "
                           f"azimuth {row['vaa']:.1f} deg", fontsize=9)
            axis.set_xlabel("pixel column")
            axis.set_ylabel("pixel row")
            figure.text(0.5, 0.01, subtitle, ha="center", fontsize=6)
            figure.tight_layout(rect=(0, 0.03, 1, 1))
            figure.savefig(folder / name / f"{name}_theta_{row['theta']:05.1f}.png", dpi=130)
            plt.close(figure)

        columns = min(4, len(rows))
        grid_rows = int(np.ceil(len(rows) / columns))
        figure, axes = plt.subplots(grid_rows, columns, figsize=(3.0 * columns + 0.8, 3.2 * grid_rows + 0.8), squeeze=False)
        for axis in axes.ravel():
            axis.axis("off")
        for axis, row, image in zip(axes.ravel(), rows, values):
            shown = axis.imshow(image, vmin=0.0, vmax=vmax, cmap="viridis")
            axis.set_title(f"{row['theta']:g} deg (view zenith {row['vza']:.1f})", fontsize=8)
        figure.colorbar(shown, ax=axes, label=quantity, fraction=0.03, pad=0.02)
        figure.suptitle(f"{quantity} against scattering angle - {subtitle}", fontsize=8)
        figure.savefig(folder / f"{name}_all_angles.pdf")
        plt.close(figure)


def run_dataset(path: Path, base, theta_list: list[float], output_root: Path) -> None:
    config = dataclasses.replace(
        base,
        # SMART-G's 3D mode cannot hold the sea's water body (the sea then only reflects)
        water_body=False,
        cloud_source=playground_core.CLOUD_BOMEX,
        bomex_netcdf=str(path),
        pointing_noise_enabled=False,  # a curve needs the same view at every angle
        renders=1,
        **({"resolution": RESOLUTION} if RESOLUTION is not None else {}),
        **({"photons_per_pixel": PHOTONS_PER_PIXEL} if PHOTONS_PER_PIXEL is not None else {}),
        **({"band_set": BAND_SET} if BAND_SET is not None else {}),
        **({"wavelength_nm": WAVELENGTH_NM} if WAVELENGTH_NM is not None else {}),
        **({"channel": CHANNEL} if CHANNEL is not None else {}),
        **({"grid_step_nm": GRID_STEP_NM} if GRID_STEP_NM is not None else {}),
    )
    config.validate()
    if config.mode != playground_core.SINGLE_MODE:
        raise ValueError(
            f"the scenario's mode is {config.mode!r}: this script needs 'Single satellite' "
            "(a formation has a different view, so a different scattering angle, per satellite)"
        )

    print(f"Cloud (from the scenario): extinction {config.extinction_method}, "
          f"crop {config.bomex_crop_cells or 'none'}, phase function {config.phase_function}"
          + (f" (g = {config.hg_g})" if config.phase_function == playground_core.cloud_microphysics.PHASE_HG else ""))
    simulation = playground_core.PlaygroundSimulation.from_config(config)
    cloud = playground_core.resolve_cloud(config, print)  # builds the cloud file when needed
    point_a_km = simulation.point_a_km(config.off_nadir_deg, config.view_azimuth_deg)
    vza, vaa = main.viewing_angles_deg(simulation.position_km, main.aim_point_km(point_a_km, cloud.center_height_km))
    print(f"View from the ground point: zenith {vza:.2f} deg, azimuth {vaa:.2f} deg "
          f"(satellite at {simulation.time_utc})")

    sza, saa = config.sza_deg, config.saa_deg
    print(f"Sun (the scenario's): zenith {sza:.2f} deg, azimuth {saa:.2f} deg")
    plan, shadowed = [], []
    for theta in theta_list:
        vza, vaa = camera_for_scattering_angle(theta, sza, saa)
        if vza >= 90.0 - 1e-9:
            shadowed.append((theta, vza, True, "the camera would be at or below the horizon of the cloud"))
            continue
        pointing = pointing_for_view(simulation, cloud.center_height_km, vza, vaa)
        if pointing is None:
            shadowed.append((theta, vza, True, "the satellite cannot point there (past the Earth's limb from its position)"))
            continue
        eta, phi = pointing
        if eta > playground_core.MAX_OFF_NADIR_DEG:
            shadowed.append((theta, vza, False, f"it needs an off-nadir angle of {eta:.1f} deg, above the playground's "
                                                f"limit of {playground_core.MAX_OFF_NADIR_DEG:g} deg"))
            continue
        view_zenith, view_azimuth = view_from_pointing(simulation, eta, phi, cloud.center_height_km)
        check = scattering_angle_deg(sza, saa, view_zenith, view_azimuth)
        if abs(check - theta) > 1e-4:
            raise AssertionError(f"scattering angle {check} instead of {theta} for the pointing found")
        plan.append({"theta": theta, "vza": view_zenith, "vaa": view_azimuth, "eta": eta, "phi": phi})
        print(f"  scattering angle {theta:6.2f} deg: view zenith {view_zenith:5.2f}, azimuth {view_azimuth:6.2f} deg "
              f"(pointing: off-nadir {eta:5.2f}, azimuth {phi:6.2f} deg)")
    for earth, headline in (
        (True, "shadowed by the Earth (the satellite cannot see the cloud from there)"),
        (False, "outside the satellite's pointing range in the playground"),
    ):
        group = [s for s in shadowed if s[2] is earth]
        if group:
            print(
                f"  WARNING: {len(group)} of the {len(theta_list)} camera directions are {headline} and are skipped:\n    "
                + "\n    ".join(f"scattering angle {theta:g} deg (view zenith {vza:.1f} deg): {why}" for theta, vza, _, why in group)
            )
    if not plan:
        raise ValueError("no camera direction can be used (shadowed by the Earth or out of range): choose other scattering angles")

    # the margin of the simulation box (a non-periodic domain): it must hold the shift of the render's sensors
    # (the camera) and the sideways reach of the sun's rays
    margin = None
    if not config.periodic:
        top, height = grid_top_km(cloud.dat_path), cloud.center_height_km
        automatic = HORIZ_MARGIN_KM is None
        margin = automatic_margin_km(top, height, sza, max(p["vza"] for p in plan)) if automatic else float(HORIZ_MARGIN_KM)
        main.SMARTG_3D_HORIZ_EXTEND_KM = margin  # read when the scene is built; restored after the run
        print(f"  Margin of the simulation box around the cloud field: {margin:g} km "
              f"({'automatic' if automatic else 'HORIZ_MARGIN_KM'}; main.py's default is {DEFAULT_MARGIN_KM:g} km)")
        limit = camera_limit_deg(margin, top, height)
        too_far = [p for p in plan if p["vza"] >= limit]
        if too_far:
            names = ", ".join(f"{p['theta']:g}" for p in too_far)
            print(
                f"  WARNING: {len(too_far)} of the {len(plan)} camera directions need a view zenith of {limit:.1f} deg or more "
                f"({names} deg scattering angle): with a margin of {margin:g} km the render's sensors would leave the "
                "simulation box, and they are skipped. Set a larger HORIZ_MARGIN_KM, or None for the automatic margin."
            )
            plan = [p for p in plan if p["vza"] < limit]
            if not plan:
                raise ValueError("no camera direction fits in the simulation box: set a larger HORIZ_MARGIN_KM")
        sun_limit = direct_sun_limit_deg(cloud.dat_path, height)
        if sza > sun_limit:
            print(
                f"  WARNING: the sun is {sza:.1f} deg from the zenith, more than about {sun_limit:.0f} deg: with a margin of "
                f"{margin:g} km the sun ray that would reach the cloud crosses the top of the grid outside the box, and in "
                "tests the cloud was then nearly black (reflectance 0.02 instead of 0.5, and the DoLP that of the air). "
                "Set a larger HORIZ_MARGIN_KM (or None for the automatic margin), or \"periodic\": true in the scenario."
            )
        else:
            print(f"  (the direct sun reaches the cloud up to a sun zenith of about {sun_limit:.0f} deg with this margin)")

    print(f"  {len(plan)} scattering angles {plan[0]['theta']:g}-{plan[-1]['theta']:g} deg: view zenith "
          f"{min(p['vza'] for p in plan):.1f} to {max(p['vza'] for p in plan):.1f} deg")
    spectrum = playground_core.config_spectrum(config)
    labels = tuple(spectrum.labels)
    most = max(len(part.wavelengths_nm) for part in spectrum.runs)
    if most > 1:
        print(f"  WARNING: a SMART-G run here has {most} wavelengths. In the tests made for run_cloudbow_voxel_headless.py "
              "such runs lost the cloud's Mie phase function (and so the bow). Use one wavelength: BAND_SET = \"single\".")
    if len(labels) != 1:
        raise ValueError(f"this script makes images of one channel, and the scenario has {len(labels)} ({labels}): "
                         "set BAND_SET = \"single\" (or a CHANNEL)")
    if PLAN_ONLY:
        return

    folder = output_root / path.name.split(".")[0]
    # The longest file written for an angle has a name of about 150 characters; Windows refuses
    # paths over 260, and says only "no such file" after the (long) render.
    longest = len(str(folder / "theta_000.0")) + 1 + 150
    if sys.platform == "win32" and longest > 255:
        raise ValueError(
            f"the output paths would be about {longest} characters long, over Windows' limit of 260: "
            f"set OUTPUT_DIR (now {output_root}) to a shorter path"
        )
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for number, step in enumerate(plan, 1):
        theta = step["theta"]
        print(f"--- angle {number}/{len(plan)}: scattering {theta:g} deg (view zenith {step['vza']:.2f}, azimuth "
              f"{step['vaa']:.2f}) ---")
        started = time.time()
        playground_core.OUTPUT_DIR = folder / f"theta_{theta:05.1f}"
        simulation.render(dataclasses.replace(config, off_nadir_deg=step["eta"], view_azimuth_deg=step["phi"]), on_status=print)
        [stokes_file] = playground_core.OUTPUT_DIR.glob("*_stokes.npy")
        rows.append({**step, **stokes_maps(np.load(stokes_file))})
        print(f"  image mean intensity {rows[-1]['mean_intensity']:.4g}, DoLP of the mean Stokes parameters "
              f"{100 * rows[-1]['mean_dolp']:.2f} %   ({time.time() - started:.0f} s)")
        write_csv(folder, rows, (sza, saa))  # after each angle
    write_heatmaps(folder, path.name.split(".")[0], labels[0], rows, (sza, saa), config)
    print(f"Images: {folder / 'intensity'}, {folder / 'dolp'}; all angles: intensity_all_angles.pdf, dolp_all_angles.pdf")


def main_script() -> int:
    config_path = resolve(SCENARIO_CONFIG)
    try:
        base = playground_core.PlaygroundConfig.load(config_path, require_dataset=False)
    except (OSError, ValueError) as exc:
        print(f"Could not load {config_path}: {exc}", file=sys.stderr)
        return 2
    if RESOLUTION is not None and RESOLUTION not in playground_core.RESOLUTION_PRESETS:
        print(f"RESOLUTION must be one of {list(playground_core.RESOLUTION_PRESETS)}", file=sys.stderr)
        return 2

    datasets = find_datasets()
    if not datasets:
        print("No BOMEX files found (see BOMEX_DIR at the top of this file).", file=sys.stderr)
        return 2

    playground_core.GENERATED_CLOUD_DIR = resolve(GENERATED_CLOUD_DIR)
    playground_core.SAVE_RADIANCE = False
    output_root = resolve(OUTPUT_DIR)
    theta_list = angles()
    print(f"{len(datasets)} BOMEX file(s); {len(theta_list)} scattering angles "
          f"({theta_list[0]:g} to {theta_list[-1]:g} deg)")

    done, skipped, failed = [], [], []
    try:
        for number, path in enumerate(datasets, 1):
            print(f"\n===== [{number}/{len(datasets)}] {path.name} =====")
            try:
                if not path.is_file():
                    raise FileNotFoundError(f"{path} does not exist")
                if not has_droplet_spectrum(path):
                    print("No FNCD droplet spectrum in this file: skipped.")
                    skipped.append(path.name)
                    continue
                run_dataset(path, base, theta_list, output_root)
                done.append(path.name)
            except Exception:  # one bad file should not stop the rest of the batch
                traceback.print_exc()
                failed.append(path.name)
    finally:
        main.SMARTG_3D_HORIZ_EXTEND_KM = DEFAULT_MARGIN_KM  # the margin is set per run; put main.py's back

    print(f"\nDone: {len(done)}, skipped: {len(skipped)}, failed: {len(failed)}")
    for name in failed:
        print(f"  failed: {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main_script())
