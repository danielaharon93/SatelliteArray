"""Degree of linear polarization against scattering angle, for BOMEX clouds - headlessly.

The cloudbow (the cloud's version of a rainbow) sits at scattering angles of about
135-150 degrees, and shows most clearly in the polarization. This script renders the
scenario of a saved config (the satellite, its view, the surface, the atmosphere, the
band set, the resolution, the photons...) on each BOMEX cloud once per scattering angle,
and plots the degree of linear polarization (DoLP) of the image against that angle in a PDF.

The satellite's view is fixed by the scenario, so the scattering angle is changed by moving
the SUN, in the vertical plane of the view (the sun's azimuth is the camera's), at the
zenith angle that gives each scattering angle:

    scattering angle = 180 deg - |sun zenith - view zenith|    (view zenith: at the ground point)

The general formula, which the script also prints for each angle as a check, is
cos(scattering angle) = -(cos(sza) cos(vza) + sin(sza) sin(vza) cos(saa - vaa)).
With the sun on the camera's side the sea's sun glint is not in the view.

The cloud (BOMEX files, how the extinction is computed, the crop, the Mie phase function)
is that of run_bomex_headless.py: its SETTINGS (BOMEX_DIR, BOMEX_PATTERN, BOMEX_FILES,
SCENARIO_CONFIG, EXTINCTION_METHOD, CROP_CELLS) are used, so they are set in one place.
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

# One folder per dataset, in it one folder per angle (the images, as in run_bomex_headless.py),
# and cloudbow_dolp.pdf and cloudbow_dolp.csv.
OUTPUT_DIR = "cloudbow_runs"
# =======================================================================

# The largest sun zenith angle the plan may need (the sun nearly at the horizon is not a
# useful, or well simulated, case).
MAX_SUN_ZENITH_DEG = 80.0

import run_bomex_headless as bomex  # noqa: E402  (does the no-display setup of run_from_config.py)

playground_core = bomex.playground_core
main = playground_core.main


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


def polarization_of(stokes: np.ndarray) -> dict:
    """(4, h, w, n_channels) Stokes parameters -> per channel the mean Stokes parameters over the
    image, the DoLP of those means (the intensity-weighted one: bright cloud pixels count most, and
    the noise of single pixels, which biases a pixel's DoLP high, averages out), the median of the
    pixels' own DoLPs, and the AoLP of the mean."""
    mean = stokes.reshape(4, -1, stokes.shape[-1]).mean(axis=1)  # (4, n_channels)
    dolp, aolp = playground_core.stokes_to_polarization(mean)
    pixel_dolp, _ = playground_core.stokes_to_polarization(stokes)
    return {
        "stokes_mean": mean,
        "dolp": np.asarray(dolp),
        "aolp": np.asarray(aolp),
        "dolp_pixel_median": np.median(pixel_dolp.reshape(-1, stokes.shape[-1]), axis=0),
    }


def write_outputs(folder: Path, dataset: str, labels: tuple, rows: list[dict], view: tuple, config) -> None:
    """cloudbow_dolp.csv (one line per angle and channel) and cloudbow_dolp.pdf."""
    with (folder / "cloudbow_dolp.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["scattering_angle_deg", "sun_zenith_deg", "sun_azimuth_deg", "channel", "I", "Q", "U", "V",
             "dolp_percent", "dolp_percent_pixel_median", "aolp_deg"]
        )
        for row in rows:
            for k, label in enumerate(labels):
                s = row["stokes_mean"][:, k]
                writer.writerow(
                    [f"{row['theta']:.3f}", f"{row['sza']:.3f}", f"{row['saa']:.3f}", label,
                     *(f"{v:.6g}" for v in s), f"{100 * row['dolp'][k]:.4f}",
                     f"{100 * row['dolp_pixel_median'][k]:.4f}", f"{row['aolp'][k]:.2f}"]
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
    axis.set_ylabel("DoLP of the image (%)")
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
        "DoLP = sqrt(Q^2 + U^2) / I of the image's mean Stokes parameters (bright pixels count most).",
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
        extinction_method=bomex.EXTINCTION_METHOD,
        bomex_crop_cells=bomex.CROP_CELLS,
        phase_function=bomex.PHASE_FUNCTION,
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
        rows.append({"theta": theta, "sza": sza, "saa": saa, **polarization_of(stokes)})
        print("  DoLP " + ", ".join(f"{label}: {100 * d:.2f} %" for label, d in zip(labels, rows[-1]["dolp"]))
              + f"   ({time.time() - started:.0f} s)")
        write_outputs(folder, path.name.split(".")[0], labels, rows, (vza, vaa), config)  # after each angle
    print(f"PDF: {folder / 'cloudbow_dolp.pdf'}")


def main_script() -> int:
    config_path = bomex.resolve(bomex.SCENARIO_CONFIG)
    try:
        base = playground_core.PlaygroundConfig.load(config_path, require_dataset=False)
    except (OSError, ValueError) as exc:
        print(f"Could not load {config_path}: {exc}", file=sys.stderr)
        return 2
    if RESOLUTION is not None and RESOLUTION not in playground_core.RESOLUTION_PRESETS:
        print(f"RESOLUTION must be one of {list(playground_core.RESOLUTION_PRESETS)}", file=sys.stderr)
        return 2

    datasets = bomex.find_datasets()
    if not datasets:
        print("No BOMEX files found (see BOMEX_DIR in run_bomex_headless.py).", file=sys.stderr)
        return 2

    playground_core.GENERATED_CLOUD_DIR = bomex.resolve(bomex.GENERATED_CLOUD_DIR)
    playground_core.SAVE_RADIANCE = False
    output_root = bomex.resolve(OUTPUT_DIR)
    theta_list = angles()
    print(f"{len(datasets)} BOMEX file(s); {len(theta_list)} scattering angles "
          f"({theta_list[0]:g} to {theta_list[-1]:g} deg)")

    done, skipped, failed = [], [], []
    for number, path in enumerate(datasets, 1):
        print(f"\n===== [{number}/{len(datasets)}] {path.name} =====")
        try:
            if not path.is_file():
                raise FileNotFoundError(f"{path} does not exist")
            if not bomex.has_droplet_spectrum(path):
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
