"""Degree of linear polarization against scattering angle, for BOMEX clouds - headlessly.

The cloudbow (the cloud's version of a rainbow) sits at scattering angles of about
135-150 degrees, and shows most clearly in the polarization. This script renders the
scenario of a saved config (the satellite, its view, the surface, the atmosphere, the
band set, the resolution, the photons...) on each BOMEX cloud once per scattering angle,
and plots the degree of linear polarization (DoLP) against that angle in a PDF. The DoLP is
that of a small patch of pixels (PATCH_SIZE_PX, 3x3 by default) that look at thick cloud: the
block whose thinnest pixel has the largest optical depth along its line of sight. The sea's
pixels are not part of it.

The satellite's view is fixed by the scenario, so the scattering angle is changed by moving
the SUN, in the vertical plane of the view (the sun's azimuth is the camera's), at the
zenith angle that gives each scattering angle:

    scattering angle = 180 deg - |sun zenith - view zenith|    (view zenith: at the ground point)

The general formula, which the script also prints for each angle as a check, is
cos(scattering angle) = -(cos(sza) cos(vza) + sin(sza) sin(vza) cos(saa - vaa)).
With the sun on the camera's side the sea's sun glint is not in the view.

The cloud's physics - how the extinction is computed (extinction_method), the crop
(bomex_crop_cells) and the droplet phase function (phase_function, hg_g) - is the
scenario's own, from the config file: set it there (in the GUI, "Save configuration").
This script only chooses which BOMEX files to run and where things go (SETTINGS below).
The pointing error is switched off here (a curve needs the same view at every angle).

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
BAND_SET: str | None = None
GRID_STEP_NM: float | None = None

# True: only print the sun directions and scattering angles of the plan (no render; needs
# no GPU and no SMART-G).
PLAN_ONLY = False

# The DoLP is taken from a small patch of pixels that look at thick cloud, not from the whole image
# (whose sea pixels would dilute it). The patch is the PATCH_SIZE_PX x PATCH_SIZE_PX block (an odd
# number; 1 is a single pixel) whose thinnest pixel has the largest optical depth along its line of
# sight; the DoLP is that of the patch's mean Stokes parameters. It is the same block at every angle
# (only the sun moves). The run stops if even the best block has an optical depth below
# MIN_PATCH_OPTICAL_DEPTH in some pixel (the line of sight may then reach the sea).
PATCH_SIZE_PX = 3
MIN_PATCH_OPTICAL_DEPTH = 5.0

# One folder per dataset, in it one folder per angle (the images, as in run_bomex_headless.py),
# and cloudbow_dolp.pdf and cloudbow_dolp.csv.
OUTPUT_DIR = "cloudbow_runs"
# =======================================================================

# The largest sun zenith angle the plan may need (the sun nearly at the horizon is not a
# useful, or well simulated, case).
MAX_SUN_ZENITH_DEG = 80.0

here = Path(__file__).parent


def resolve(path: str) -> Path:
    return (here / path).expanduser()  # an absolute path wins


# The no-display setup (Traits backend, SMARTG_DIR_AUXDATA, the import workaround for main.py)
# lives in run_from_config.py; importing it only does that setup and does not run anything.
import run_from_config  # noqa: E402

playground_core = run_from_config.playground_core
main = playground_core.main


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


def sun_for_scattering_angle(theta_deg: float, vza_deg: float, vaa_deg: float) -> tuple[float, float]:
    """(sun zenith, sun azimuth) that give the scattering angle for the view (vza, vaa): the sun
    in the vertical plane of the view, on the camera's side, `180 - theta` degrees from the view
    direction (farther from the zenith than the camera)."""
    return vza_deg + (180.0 - theta_deg), vaa_deg % 360.0


def angles() -> list[float]:
    if ANGLE_STEP_DEG <= 0 or ANGLE_STOP_DEG < ANGLE_START_DEG:
        sys.exit("Set ANGLE_START_DEG <= ANGLE_STOP_DEG and ANGLE_STEP_DEG > 0.")
    n = int(np.floor((ANGLE_STOP_DEG - ANGLE_START_DEG) / ANGLE_STEP_DEG + 1e-9)) + 1
    return [float(ANGLE_START_DEG + k * ANGLE_STEP_DEG) for k in range(n)]


def pixel_optical_depths(
    dat_path: Path, width_px: int, height_px: int, vza_deg: float, vaa_deg: float,
    target_height_km: float, periodic: bool,
) -> np.ndarray:
    """(height_px, width_px) optical depth along each pixel's line of sight through the cloud file.
    The pixels are laid out as in playground_core.footprint_sensors(): points on a regular grid
    over the field, all looking along one direction, the lines of sight crossing the field's
    centre at `target_height_km` (the same geometry as playground_core.pixel_voxel_statistics()).
    The extinction is the cloud file's, at its reference wavelength."""
    from smartg.atmosphere import read_i3rc_cloud

    field = read_i3rc_cloud(dat_path)
    ext = field["ext"].values  # (nz, ny, nx), 1/km
    xb, yb, zb = (np.asarray(field[name].values, dtype=float) for name in ("x_bounds", "y_bounds", "z_bounds"))
    nz, ny, nx = ext.shape
    layers = np.flatnonzero((ext > 0).any(axis=(1, 2)))
    if layers.size == 0:
        raise ValueError("the cloud has no cloudy voxel")
    z_lo, z_hi = float(zb[layers[0]]), float(zb[layers[-1] + 1])

    theta, travel = np.radians(vza_deg), np.radians((vaa_deg + 180.0) % 360.0)
    shift_x, shift_y = np.tan(theta) * np.cos(travel), np.tan(theta) * np.sin(travel)  # km per km of descent
    x0, x1, y0, y1 = xb[0], xb[-1], yb[0], yb[-1]
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    pitch_x, pitch_y = (x1 - x0) / width_px, (y1 - y0) / height_px
    px, py = np.meshgrid(
        x0 + (np.arange(width_px) + 0.5) * pitch_x - cx, y0 + (np.arange(height_px) + 0.5) * pitch_y - cy
    )

    step_km = 0.25 * float(np.diff(zb[layers[0] : layers[-1] + 2]).min())
    heights = z_hi - (np.arange(int(np.ceil((z_hi - z_lo) / step_km))) + 0.5) * step_km
    path_km = step_km / np.cos(theta)  # length of the line of sight in each height step
    depth = np.zeros(px.shape)
    for height in heights:
        x = cx + px + (target_height_km - height) * shift_x
        y = cy + py + (target_height_km - height) * shift_y
        if periodic:
            x = x0 + (x - x0) % (x1 - x0)
            y = y0 + (y - y0) % (y1 - y0)
        ix = np.searchsorted(xb, x, side="right") - 1
        iy = np.searchsorted(yb, y, side="right") - 1
        iz = int(np.searchsorted(zb, height, side="right")) - 1
        inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        depth += np.where(inside, ext[iz, np.clip(iy, 0, ny - 1), np.clip(ix, 0, nx - 1)], 0.0) * path_km
    return depth


def choose_patch(depth: np.ndarray, size: int) -> tuple[int, int, float]:
    """The size x size block of pixels (top row, left column) whose thinnest pixel has the largest
    optical depth - so that every pixel of the block looks at thick cloud - and that thinnest
    optical depth."""
    if size < 1 or size % 2 == 0:
        raise ValueError(f"PATCH_SIZE_PX must be an odd number, 1 or more (it is {size})")
    height, width = depth.shape
    if size > min(height, width):
        raise ValueError(f"PATCH_SIZE_PX {size} is larger than the image ({width}x{height})")
    thinnest = np.min(
        [depth[r : height - size + 1 + r, c : width - size + 1 + c] for r in range(size) for c in range(size)],
        axis=0,
    )
    row, col = np.unravel_index(np.argmax(thinnest), thinnest.shape)
    return int(row), int(col), float(thinnest[row, col])


def polarization_of(stokes: np.ndarray, row: int, col: int, size: int) -> dict:
    """(4, h, w, n_channels) Stokes parameters -> per channel the mean Stokes parameters over the
    size x size patch of pixels whose top left pixel is (row, col), the DoLP of those means (the
    noise of single pixels, which biases a pixel's own DoLP high, averages out) and their AoLP."""
    patch = stokes[:, row : row + size, col : col + size, :]
    mean = patch.reshape(4, -1, stokes.shape[-1]).mean(axis=1)  # (4, n_channels)
    dolp, aolp = playground_core.stokes_to_polarization(mean)
    return {"stokes_mean": mean, "dolp": np.asarray(dolp), "aolp": np.asarray(aolp)}


def write_outputs(
    folder: Path, dataset: str, labels: tuple, rows: list[dict], view: tuple, config, patch: tuple
) -> None:
    """cloudbow_dolp.csv (one line per angle and channel) and cloudbow_dolp.pdf. `patch` is
    (top row, left column, size, thinnest optical depth)."""
    patch_row, patch_col, size, patch_depth = patch
    with (folder / "cloudbow_dolp.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["scattering_angle_deg", "sun_zenith_deg", "sun_azimuth_deg", "channel", "I", "Q", "U", "V",
             "dolp_percent", "aolp_deg", "patch_top_row", "patch_left_column", "patch_size_px",
             "patch_min_optical_depth"]
        )
        for row in rows:
            for k, label in enumerate(labels):
                s = row["stokes_mean"][:, k]
                writer.writerow(
                    [f"{row['theta']:.3f}", f"{row['sza']:.3f}", f"{row['saa']:.3f}", label,
                     *(f"{v:.6g}" for v in s), f"{100 * row['dolp'][k]:.4f}", f"{row['aolp'][k]:.2f}",
                     patch_row, patch_col, size, f"{patch_depth:.2f}"]
                )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"blue": "tab:blue", "green": "tab:green", "red": "tab:red"}
    if len(labels) == 3:
        line_colours = [colours["blue"], colours["green"], colours["red"]]
    else:
        line_colours = [colours.get(config.channel, "black")] * len(labels)
    theta = np.array([row["theta"] for row in rows])
    figure, axis = plt.subplots(figsize=(7.5, 4.6))
    for k, label in enumerate(labels):
        values = 100 * np.array([row["dolp"][k] for row in rows])
        axis.plot(theta, values, "o-", color=line_colours[k], label=label)
    axis.set_xlabel("Scattering angle (deg)")
    axis.set_ylabel("DoLP of the cloud patch (%)")
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8, loc="best")
    vza, vaa = view
    figure.suptitle("Degree of linear polarization against scattering angle", fontsize=11)
    axis.set_title(
        f"{dataset}\n"
        f"view zenith {vza:.1f} deg, azimuth {vaa:.1f} deg; sun in the view's vertical plane, "
        f"zenith {rows[0]['sza']:.1f} to {rows[-1]['sza']:.1f} deg\n"
        f"{config.band_set}, {config.resolution}, {config.photons_per_pixel:g} photons/pixel",
        fontsize=7,
    )
    figure.text(
        0.5, 0.005,
        f"DoLP = sqrt(Q^2 + U^2) / I of the mean Stokes parameters of a {size}x{size} pixel patch "
        f"(row {patch_row}, column {patch_col}) whose lines of sight all cross cloud of optical depth >= {patch_depth:.1f}.",
        ha="center", fontsize=7,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 0.97))
    figure.savefig(folder / "cloudbow_dolp.pdf")
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

    plan = []
    for theta in theta_list:
        sza, saa = sun_for_scattering_angle(theta, vza, vaa)
        if not 0.0 <= sza <= MAX_SUN_ZENITH_DEG:
            raise ValueError(
                f"scattering angle {theta:g} deg needs a sun zenith angle of {sza:.1f} deg with this view "
                f"(allowed 0 to {MAX_SUN_ZENITH_DEG:g}): choose other angles"
            )
        check = scattering_angle_deg(sza, saa, vza, vaa)
        if abs(check - theta) > 1e-6:
            raise AssertionError(f"scattering angle {check} instead of {theta}")
        plan.append((theta, sza, saa))
        print(f"  scattering angle {theta:6.2f} deg: sun zenith {sza:6.2f}, azimuth {saa:6.2f} deg")

    width_px, height_px = playground_core.RESOLUTION_PRESETS[config.resolution]
    depth = pixel_optical_depths(
        cloud.dat_path, width_px, height_px, vza, vaa, cloud.center_height_km, config.periodic
    )
    patch_row, patch_col, patch_depth = choose_patch(depth, PATCH_SIZE_PX)
    print(f"Patch of {PATCH_SIZE_PX}x{PATCH_SIZE_PX} pixels at row {patch_row}, column {patch_col}: "
          f"optical depth along its lines of sight at least {patch_depth:.1f} "
          f"(the image's largest: {depth.max():.1f}; {100 * np.mean(depth >= MIN_PATCH_OPTICAL_DEPTH):.0f} % "
          f"of the pixels reach {MIN_PATCH_OPTICAL_DEPTH:g})")
    if patch_depth < MIN_PATCH_OPTICAL_DEPTH:
        raise ValueError(
            f"even the best {PATCH_SIZE_PX}x{PATCH_SIZE_PX} patch has a pixel of optical depth {patch_depth:.1f}, "
            f"below MIN_PATCH_OPTICAL_DEPTH ({MIN_PATCH_OPTICAL_DEPTH:g}): lower it or use a smaller patch"
        )
    patch = (patch_row, patch_col, PATCH_SIZE_PX, patch_depth)
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
    labels = tuple(playground_core.config_spectrum(config).labels)
    rows = []
    for number, (theta, sza, saa) in enumerate(plan, 1):
        print(f"--- angle {number}/{len(plan)}: scattering {theta:g} deg (sun zenith {sza:.2f}, azimuth {saa:.2f}) ---")
        started = time.time()
        playground_core.OUTPUT_DIR = folder / f"theta_{theta:05.1f}"
        simulation.render(dataclasses.replace(config, sza_deg=sza, saa_deg=saa), on_status=print)
        [stokes_file] = playground_core.OUTPUT_DIR.glob("*_stokes.npy")
        stokes = np.load(stokes_file)
        rows.append({"theta": theta, "sza": sza, "saa": saa, **polarization_of(stokes, patch_row, patch_col, PATCH_SIZE_PX)})
        print("  DoLP " + ", ".join(f"{label}: {100 * d:.2f} %" for label, d in zip(labels, rows[-1]["dolp"]))
              + f"   ({time.time() - started:.0f} s)")
        write_outputs(folder, path.name.split(".")[0], labels, rows, (vza, vaa), config, patch)  # after each angle
    print(f"PDF: {folder / 'cloudbow_dolp.pdf'}")


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

    print(f"\nDone: {len(done)}, skipped: {len(skipped)}, failed: {len(failed)}")
    for name in failed:
        print(f"  failed: {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main_script())
