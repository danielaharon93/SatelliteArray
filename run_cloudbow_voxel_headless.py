"""Degree of linear polarization against scattering angle for ONE HOMOGENEOUS VOXEL - headlessly.

A simple case for finding the cloudbow before trying realistic clouds. It is run_cloudbow_headless.py
with a different cloud: instead of a whole BOMEX field, the cloud is a single voxel of it, so the
medium is homogeneous (one extinction coefficient, one droplet effective radius). The voxel is taken
from the BOMEX file: of the voxels whose optical depth (extinction x the voxel's thickness) is at
least MIN_VOXEL_OPTICAL_DEPTH - the cores of the clouds, not their hazy edges or the clear air - the
one whose effective radius is the median. It is written as a small cloud file (the layers below it
empty, as in the BOMEX cloud file, so the atmosphere is still modelled all the way down).

With PERIODIC the voxel repeats horizontally, which makes it an infinite homogeneous layer as thick as
the voxel (20 m): every pixel of the image then sees the same medium, so the image's mean Stokes
parameters are the right quantity (no sea or cloud-edge pixels to separate) and the pixels only add
photons. (SMART-G's periodic grid needs at least 2 cells per direction, so the file holds a 2 x 2
block of identical voxels.) Without PERIODIC, the voxel is one isolated 20 m cube, which is mostly
transparent to a camera that looks at the sea behind it.

As in run_cloudbow_headless.py, the satellite's view is the scenario's and the scattering angle is
changed by moving the SUN in the vertical plane of the view:

    scattering angle = 180 deg - |sun zenith - view zenith|    (view zenith: at the cloud's height)

and everything else - the satellite, the surface, the atmosphere, the resolution, the photons, the
extinction method and the phase function - is the scenario's own (configs/scenario.json). The exception
is the wavelength: BAND_SET and CHANNEL below select ONE wavelength (see there for why).

The plot also shows the SINGLE-SCATTERING polarization of the same droplets, |F12| / F11 of SMART-G's
water-cloud table (clouds/wc_sol.nc) at the voxel's effective radius and the band's wavelengths: where
the bow would be if each photon scattered once. The rendered DoLP follows it where single scattering
dominates and falls below it as multiple scattering and the air and the sea add unpolarized light.
Needs a CUDA GPU and SMART-G's auxiliary data, as run_cloudbow_headless.py does, unless PLAN_ONLY is
True. Run:

    python run_cloudbow_voxel_headless.py
"""

import csv
import dataclasses
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# ============================== SETTINGS ==============================
# Where the BOMEX netCDF files are: the folder, and which files in it (the pattern is searched in the
# folder only; use "**/*.com3D_int_2.nc" to include its subfolders). Files without the FNCD droplet
# spectrum are skipped with a message. A relative path is taken relative to this file's folder; an
# absolute path also works.
BOMEX_DIR = "/data/bomex"
BOMEX_PATTERN = "*.com3D_int_2.nc"
# Or name the files yourself (full paths); when not empty, BOMEX_DIR and BOMEX_PATTERN are ignored.
BOMEX_FILES: list[str] = []

# The saved scenario (saved from the GUI with "Save configuration"): the satellite and its view, the
# surface, the atmosphere, the band set, the resolution, the photons, and the cloud's physics
# (extinction method, phase function). Its cloud source and BOMEX file settings are replaced here.
SCENARIO_CONFIG = "configs/scenario.json"

# Where the generated cloud files go (the BOMEX field's, reused on later runs, and the voxel's).
GENERATED_CLOUD_DIR = "generated_clouds"

# The voxel. It is chosen from the whole BOMEX field (the scenario's crop is not used) among the voxels
# whose optical depth, extinction x thickness, is at least MIN_VOXEL_OPTICAL_DEPTH (the median voxel
# is 0.65, the top 1 % are above 3.7): the one with the median effective radius of those. Or name it:
# VOXEL = (ix, iy, iz), zero-based indices into the field, which overrides the choice.
MIN_VOXEL_OPTICAL_DEPTH = 3.0
VOXEL: tuple[int, int, int] | None = None
# True: the voxel repeats horizontally (an infinite homogeneous layer, the voxel's thickness). False:
# one isolated voxel. This is part of what makes the medium homogeneous, so it is set here and not by
# the scenario's own "periodic" setting.
PERIODIC = True

# The scattering angles, in degrees: from START to STOP (included) every STEP. The primary bow of
# droplets like these is a few degrees wide, near 140 degrees, so a step of 1 or less is needed.
ANGLE_START_DEG = 125.0
ANGLE_STOP_DEG = 155.0
ANGLE_STEP_DEG = 1.0

# None keeps the scenario's value. Every pixel sees the same medium, so a smaller image with more
# photons per pixel is the same experiment; the DoLP is that of the image's mean Stokes parameters.
# RESOLUTION is a name from playground_core.RESOLUTION_PRESETS, e.g. "Small (32x32)".
RESOLUTION: str | None = None
PHOTONS_PER_PIXEL: float | None = None
# The wavelengths. IMPORTANT: in the tests made for this script, a SMART-G run with MORE THAN ONE
# wavelength gave the voxel the same result with the Mie and the Henyey-Greenstein phase function (no
# bow, and the DoLP 5 % instead of 31 % at 142 deg), while a run with ONE wavelength gave the bow; so
# this script uses one wavelength by default: the "modis" band set (one REPTRAN wavelength per channel)
# with the scenario's single channel (CHANNEL: "red" is 646 nm). A run with several wavelengths prints a
# warning. None keeps the scenario's value. GRID_STEP_NM is for the grid band sets.
BAND_SET: str | None = "modis"
CHANNEL: str | None = None
GRID_STEP_NM: float | None = None

# True: only choose the voxel, write its file and print the plan and the single-scattering bow (no
# render; needs no GPU).
PLAN_ONLY = False

# One folder per dataset, in it one folder per angle, and cloudbow_voxel.pdf, cloudbow_voxel.csv and
# voxel.json (which voxel it was).
OUTPUT_DIR = "cloudbow_voxel_runs"
# =======================================================================

# The largest sun zenith angle the plan may need.
MAX_SUN_ZENITH_DEG = 80.0
# One droplet radius needs no coarse rounding (the grid band sets round to 1 um to save memory with a
# whole cloud's many radii): keep this many decimals of the voxel's effective radius.
REFF_DECIMALS = 2
# The pixels are split into this many interleaved groups to estimate the noise of the DoLP.
NOISE_GROUPS = 16
# The periodic layer is a block of this many x this many identical voxels (2 is the smallest SMART-G's
# periodic grid works with: with one cell in a direction the image is almost black).
PERIODIC_CELLS = 2

here = Path(__file__).parent


def resolve(path: str) -> Path:
    return (here / path).expanduser()  # an absolute path wins


# The no-display setup (Traits backend, SMARTG_DIR_AUXDATA, the import workaround for main.py) lives in
# run_from_config.py; importing it only does that setup and does not run anything.
import run_from_config  # noqa: E402

playground_core = run_from_config.playground_core
main = playground_core.main
playground_core.GRID_REFF_ACC = REFF_DECIMALS  # read when the band set is built, below


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


# ---------------------------------------------------------------------------------- the geometry


def scattering_angle_deg(sza_deg: float, saa_deg: float, vza_deg: float, vaa_deg: float) -> float:
    """The angle between the light's direction before the scattering (from the sun, down) and after
    it (toward the camera): 180 deg is the exact backscatter. All four angles are in SMART-G's
    local frame (zenith from straight up, azimuth counterclockwise from east, of the direction
    toward the sun and toward the camera)."""
    sza, vza, dphi = np.radians(sza_deg), np.radians(vza_deg), np.radians(saa_deg - vaa_deg)
    cos_angle = np.cos(sza) * np.cos(vza) + np.sin(sza) * np.sin(vza) * np.cos(dphi)
    return float(np.degrees(np.arccos(np.clip(-cos_angle, -1.0, 1.0))))


def sun_for_scattering_angle(theta_deg: float, vza_deg: float, vaa_deg: float) -> tuple[float, float]:
    """(sun zenith, sun azimuth) that give the scattering angle for the view (vza, vaa): the sun in the
    vertical plane of the view, on the camera's side, `180 - theta` degrees from the view direction
    (farther from the zenith than the camera)."""
    return vza_deg + (180.0 - theta_deg), vaa_deg % 360.0


def angles() -> list[float]:
    if ANGLE_STEP_DEG <= 0 or ANGLE_STOP_DEG < ANGLE_START_DEG:
        sys.exit("Set ANGLE_START_DEG <= ANGLE_STOP_DEG and ANGLE_STEP_DEG > 0.")
    n = int(np.floor((ANGLE_STOP_DEG - ANGLE_START_DEG) / ANGLE_STEP_DEG + 1e-9)) + 1
    return [float(ANGLE_START_DEG + k * ANGLE_STEP_DEG) for k in range(n)]


# ---------------------------------------------------------------------------------- the voxel


def phase_table():
    """SMART-G's water-cloud table (clouds/wc_sol.nc, "wc" in main.SMARTG_CLOUD_FNAME): the phase matrix
    by droplet effective radius, wavelength and scattering angle, the droplets' effective variance 0.1."""
    import xarray as xr

    path = Path(os.environ["SMARTG_DIR_AUXDATA"]) / "clouds" / "wc_sol.nc"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found (SMART-G's water-cloud table)")
    return xr.open_dataset(path)


def pick_voxel(field, min_optical_depth: float, reff_range_um: tuple[float, float]):
    """((iz, iy, ix), info) of the voxel of the cloud `field` (smartg's read_i3rc_cloud) chosen as the
    docstring says: among the voxels with optical depth >= min_optical_depth (and an effective radius
    inside the table's range), the one with the median effective radius."""
    ext, reff = field["ext"].values, field["reff"].values
    thickness = np.diff(field["z_bounds"].values)[: ext.shape[0]]
    optical_depth = ext * thickness[:, None, None]
    ok = (ext > 0) & (optical_depth >= min_optical_depth) & (reff >= reff_range_um[0]) & (reff <= reff_range_um[1])
    candidates = np.argwhere(ok)
    if len(candidates) == 0:
        raise ValueError(
            f"no voxel has an optical depth of at least {min_optical_depth:g} (the largest is "
            f"{optical_depth.max():.2f}): lower MIN_VOXEL_OPTICAL_DEPTH"
        )
    radii = reff[tuple(candidates.T)]
    index = tuple(int(i) for i in candidates[np.argsort(radii, kind="stable")[len(candidates) // 2]])
    return index, {"candidates": int(len(candidates)), "candidate_reff_um": [float(radii.min()), float(radii.max())]}


def voxel_info(field, index: tuple[int, int, int]) -> dict:
    iz, iy, ix = index
    z_lo, z_hi = (float(v) for v in field["z_bounds"].values[iz : iz + 2])
    ext, reff = float(field["ext"].values[index]), float(field["reff"].values[index])
    if not ext > 0:
        raise ValueError(f"voxel (ix, iy, iz) = ({ix}, {iy}, {iz}) holds no cloud (extinction 0)")
    return {
        "ix": ix, "iy": iy, "iz": iz, "z_bottom_km": z_lo, "z_top_km": z_hi, "thickness_km": z_hi - z_lo,
        "extinction_per_km": ext, "optical_depth": ext * (z_hi - z_lo), "reff_um": reff,
    }


def write_voxel_cloud(field, info: dict, source: str, path: Path, cells: int) -> Path:
    """The cumulus-format cloud file of the voxel: `cells` x `cells` identical voxels horizontally (one
    voxel, or a block of them that a periodic domain repeats), the BOMEX field's own layers from the
    ground up to the voxel (empty but for it), and the empty padding above that the BOMEX file has (the
    atmosphere is modelled only on the grid's layers)."""
    import cumulus_file

    iz = info["iz"]
    z_bounds = field["z_bounds"].values[: iz + 2]
    pad = [e for e in cumulus_file.TOP_PAD_EDGES_KM if e > z_bounds[-1] + cumulus_file.TOP_PAD_MIN_KM]
    dx = float(np.diff(field["x_bounds"].values)[0])
    dy = float(np.diff(field["y_bounds"].values)[0])
    grid = cumulus_file.CumulusGrid(cells, cells, dx, dy, np.concatenate([z_bounds, pad]), nz_data=iz + 1)
    cf = cumulus_file.create_cumulus_file(path, grid, f"{cells}x{cells} identical voxels of {source}: {info}")
    ext = np.zeros((iz + 1, cells, cells))
    reff = np.zeros((iz + 1, cells, cells))
    ext[iz], reff[iz] = info["extinction_per_km"], info["reff_um"]
    cumulus_file.set_extinction(cf, ext)
    cumulus_file.set_effective_radius(cf, reff)
    cumulus_file.write_cells(cf)
    cumulus_file.verify_cumulus_file(cf)  # reads it back with SMART-G
    return path


# ---------------------------------------------------------------------------------- the reference


def single_scattering(spectrum, reff_um: float, theta_deg: np.ndarray) -> dict:
    """What one scattering would give, for the droplets of the voxel and the band's wavelengths: per
    channel, F11 (the phase function, normalised so that its mean over the sphere is 1) and the degree of
    linear polarization |F12| / F11 of unpolarized light, at the scattering angles `theta_deg`. A
    channel's wavelengths are combined with its weights (response x solar irradiance x bin width)."""
    import xarray as xr

    table = phase_table()
    wavelength = spectrum.wavelengths_nm
    weight = spectrum.weight if spectrum.weight is not None else np.ones(len(wavelength))
    group = spectrum.group if spectrum.group is not None else np.arange(len(wavelength))
    wavelength_range = (float(table["wav"].min()), float(table["wav"].max()))
    phase = table["phase"].interp(
        reff=min(max(reff_um, float(table["reff"].min())), float(table["reff"].max())),
        wav=xr.DataArray(np.clip(wavelength, *wavelength_range), dims="w"),
    )
    table_theta = table["theta"].values
    f11 = np.array([np.interp(theta_deg, table_theta, phase.sel(stk=0).values[i]) for i in range(len(wavelength))])
    f12 = np.array([np.interp(theta_deg, table_theta, phase.sel(stk=1).values[i]) for i in range(len(wavelength))])
    out = {"f11": [], "dolp": []}
    for k in range(len(spectrum.labels)):
        mine = group == k
        w = weight[mine, None]
        out["f11"].append((w * f11[mine]).sum(axis=0) / w.sum())
        out["dolp"].append(np.abs((w * f12[mine]).sum(axis=0)) / (w * f11[mine]).sum(axis=0))
    return {key: np.array(value) for key, value in out.items()}  # (n_channels, n_angles)


def describe_bow(theta: np.ndarray, dolp: np.ndarray, f11: np.ndarray) -> str:
    inside = (theta >= 120.0) & (theta <= 165.0)
    return (
        f"single scattering: F11 peaks at {theta[inside][np.argmax(f11[inside])]:.1f} deg, "
        f"the DoLP at {theta[inside][np.argmax(dolp[inside])]:.1f} deg ({100 * dolp[inside].max():.0f} %)"
    )


# ---------------------------------------------------------------------------------- the polarization


def polarization_of(stokes: np.ndarray) -> dict:
    """(4, h, w, n_channels) Stokes parameters -> per channel the mean Stokes parameters over the image (every
    pixel sees the same medium), the DoLP and AoLP of those means, and the standard error of the DoLP:
    the pixels are split into NOISE_GROUPS interleaved groups, each group's DoLP is taken, and the
    error is the spread of those over the square root of their number."""
    n_channels = stokes.shape[-1]
    pixels = stokes.reshape(4, -1, n_channels)
    mean = pixels.mean(axis=1)
    dolp, aolp = playground_core.stokes_to_polarization(mean)
    groups = np.array([playground_core.stokes_to_polarization(pixels[:, g::NOISE_GROUPS].mean(axis=1))[0]
                       for g in range(NOISE_GROUPS)])
    return {
        "stokes_mean": mean,
        "dolp": np.asarray(dolp),
        "aolp": np.asarray(aolp),
        "dolp_error": groups.std(axis=0, ddof=1) / np.sqrt(NOISE_GROUPS),
    }


def write_outputs(folder: Path, dataset: str, labels: tuple, rows: list[dict], reference: dict,
                  theta_reference: np.ndarray, view: tuple, config, voxel: dict) -> None:
    """cloudbow_voxel.csv (one line per angle and channel) and cloudbow_voxel.pdf."""
    with (folder / "cloudbow_voxel.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["scattering_angle_deg", "sun_zenith_deg", "sun_azimuth_deg", "channel", "I", "Q", "U", "V",
             "dolp_percent", "dolp_error_percent", "aolp_deg", "single_scattering_dolp_percent", "single_scattering_F11"]
        )
        for row in rows:
            for k, label in enumerate(labels):
                s = row["stokes_mean"][:, k]
                dolp_ss = 100 * float(np.interp(row["theta"], theta_reference, reference["dolp"][k]))
                f11_ss = float(np.interp(row["theta"], theta_reference, reference["f11"][k]))
                writer.writerow(
                    [f"{row['theta']:.3f}", f"{row['sza']:.3f}", f"{row['saa']:.3f}", label,
                     *(f"{v:.6g}" for v in s), f"{100 * row['dolp'][k]:.4f}", f"{100 * row['dolp_error'][k]:.4f}",
                     f"{row['aolp'][k]:.2f}", f"{dolp_ss:.4f}", f"{f11_ss:.6g}"]
                )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"blue": "tab:blue", "green": "tab:green", "red": "tab:red"}
    line_colours = (
        [colours["blue"], colours["green"], colours["red"]] if len(labels) == 3
        else [colours.get(config.channel, "black")] * len(labels)
    )
    theta = np.array([row["theta"] for row in rows])
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(7.5, 7.2), sharex=True, height_ratios=(3, 2))
    for k, label in enumerate(labels):
        top.errorbar(
            theta, 100 * np.array([row["dolp"][k] for row in rows]),
            yerr=100 * np.array([row["dolp_error"][k] for row in rows]),
            fmt="o-", ms=4, lw=1, capsize=2, color=line_colours[k], label=f"rendered, {label}",
        )
        top.plot(theta_reference, 100 * reference["dolp"][k], "--", color=line_colours[k], alpha=0.55,
                 label="single scattering, |F12| / F11")
        bottom.plot(theta, [row["stokes_mean"][0, k] for row in rows], "o-", ms=4, lw=1, color=line_colours[k],
                    label=f"rendered reflectance, {label}")
        twin = bottom.twinx() if k == 0 else twin
        twin.plot(theta_reference, reference["f11"][k], "--", color="0.4", alpha=0.7,
                  label="single-scattering phase function F11 (right axis)")
    top.set_ylabel("DoLP of the image (%)")
    top.set_ylim(bottom=0)
    top.grid(alpha=0.3)
    top.legend(fontsize=7, loc="best")
    bottom.set_xlabel("Scattering angle (deg)")
    bottom.set_ylabel("Mean reflectance (I)")
    bottom.grid(alpha=0.3)
    twin.set_ylabel("F11", color="0.4")
    handles = bottom.get_legend_handles_labels()
    handles2 = twin.get_legend_handles_labels()
    bottom.legend(handles[0] + handles2[0], handles[1] + handles2[1], fontsize=7, loc="best")
    vza, vaa = view
    figure.suptitle("Cloudbow of one homogeneous BOMEX voxel: polarization and reflectance against scattering angle", fontsize=10)
    top.set_title(
        f"{dataset}\n"
        f"voxel: extinction {voxel['extinction_per_km']:.1f} /km, thickness {1000 * voxel['thickness_km']:.0f} m, "
        f"optical depth {voxel['optical_depth']:.2f}, droplet radius {voxel['reff_um']:.2f} um, "
        f"{'periodic layer' if config.periodic else 'isolated voxel'}\n"
        f"view zenith {vza:.1f} deg, azimuth {vaa:.1f} deg; sun in the view's vertical plane, zenith "
        f"{rows[0]['sza']:.1f} to {rows[-1]['sza']:.1f} deg; {config.band_set}, {config.resolution}, "
        f"{config.photons_per_pixel:g} photons/pixel",
        fontsize=6.5,
    )
    figure.text(
        0.5, 0.005,
        "DoLP = sqrt(Q^2 + U^2) / I of the image's mean Stokes parameters; bars: standard error from "
        f"{NOISE_GROUPS} pixel groups.\nDashed: one scattering by the table's droplets (effective variance 0.1).",
        ha="center", fontsize=6.5,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.97))
    figure.savefig(folder / "cloudbow_voxel.pdf")
    plt.close(figure)


# ---------------------------------------------------------------------------------- the run


def run_dataset(path: Path, base, theta_list: list[float], output_root: Path) -> None:
    config = dataclasses.replace(
        base,
        # SMART-G's 3D mode cannot hold the sea's water body (the sea then only reflects)
        water_body=False,
        cloud_source=playground_core.CLOUD_BOMEX,  # valid for the config; the voxel's cloud is passed to render()
        bomex_netcdf=str(path),
        bomex_crop_cells=0,
        pointing_noise_enabled=False,  # a curve needs the same view at every angle
        renders=1,
        periodic=PERIODIC,
        **({"resolution": RESOLUTION} if RESOLUTION is not None else {}),
        **({"photons_per_pixel": PHOTONS_PER_PIXEL} if PHOTONS_PER_PIXEL is not None else {}),
        **({"band_set": BAND_SET} if BAND_SET is not None else {}),
        **({"channel": CHANNEL} if CHANNEL is not None else {}),
        **({"grid_step_nm": GRID_STEP_NM} if GRID_STEP_NM is not None else {}),
    )
    config.validate()
    if config.mode != playground_core.SINGLE_MODE:
        raise ValueError(
            f"the scenario's mode is {config.mode!r}: this script needs 'Single satellite' "
            "(a formation has a different view, so a different scattering angle, per satellite)"
        )

    # the voxel, from the whole field
    from smartg.atmosphere import read_i3rc_cloud

    field_file = playground_core.bomex_cloud_file(path, config.extinction_method, 0, on_status=print)
    field = read_i3rc_cloud(field_file)
    table = phase_table()
    reff_range = (float(table["reff"].min()), float(table["reff"].max()))
    if VOXEL is None:
        (iz, iy, ix), choice = pick_voxel(field, MIN_VOXEL_OPTICAL_DEPTH, reff_range)
        how = (f"chosen: median effective radius of the {choice['candidates']} voxels with optical depth >= "
               f"{MIN_VOXEL_OPTICAL_DEPTH:g} (their radii {choice['candidate_reff_um'][0]:.1f}-{choice['candidate_reff_um'][1]:.1f} um)")
    else:
        ix, iy, iz = VOXEL
        how = "named by VOXEL"
    voxel = voxel_info(field, (iz, iy, ix))
    if not reff_range[0] <= voxel["reff_um"] <= reff_range[1]:
        raise ValueError(f"the voxel's effective radius {voxel['reff_um']:.2f} um is outside the water-cloud table's "
                         f"{reff_range[0]:g}-{reff_range[1]:g} um")
    print(f"Voxel (ix, iy, iz) = ({ix}, {iy}, {iz}), {how}:\n"
          f"  height {voxel['z_bottom_km']:.2f}-{voxel['z_top_km']:.2f} km, extinction {voxel['extinction_per_km']:.1f} /km, "
          f"optical depth {voxel['optical_depth']:.2f}, droplet radius {voxel['reff_um']:.2f} um")
    # SMART-G's periodic grid needs at least 2 cells in each horizontal direction (one cell gives
    # almost no signal), so the periodic layer is a 2 x 2 block of identical voxels: the same medium.
    cells = PERIODIC_CELLS if config.periodic else 1
    key = hashlib.sha1(
        f"{field_file}|{iz},{iy},{ix}|{voxel['extinction_per_km']}|{voxel['reff_um']}|{cells}|v2".encode()
    ).hexdigest()[:10]
    voxel_file = resolve(GENERATED_CLOUD_DIR) / f"voxel_{key}.dat"
    if not voxel_file.exists():
        voxel_file.parent.mkdir(parents=True, exist_ok=True)
        partial = voxel_file.with_name(voxel_file.name + ".partial")
        write_voxel_cloud(field, voxel, path.name, partial, cells)
        partial.replace(voxel_file)
    cloud = playground_core.CloudSource(voxel_file, config.phase_function, config.hg_g)
    print(f"Cloud (the voxel, from the scenario's physics): extinction {config.extinction_method}, phase function "
          f"{config.phase_function}, {'periodic layer' if config.periodic else 'isolated voxel'}; file {voxel_file.name}")

    simulation = playground_core.PlaygroundSimulation.from_config(config)
    point_a_km = simulation.point_a_km(config.off_nadir_deg, config.view_azimuth_deg)
    vza, vaa = main.viewing_angles_deg(simulation.position_km, main.aim_point_km(point_a_km, cloud.center_height_km))
    print(f"View from the ground point: zenith {vza:.2f} deg, azimuth {vaa:.2f} deg (satellite at {simulation.time_utc})")

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
    print(f"  {len(plan)} scattering angles {theta_list[0]:g}-{theta_list[-1]:g} deg: sun zenith "
          f"{plan[0][1]:.1f} to {plan[-1][1]:.1f} deg, azimuth {plan[0][2]:.1f} deg")

    spectrum = playground_core.config_spectrum(config)
    labels = tuple(spectrum.labels)
    most = max(len(part.wavelengths_nm) for part in spectrum.runs)
    if most > 1:
        print(
            f"  WARNING: a SMART-G run here has {most} wavelengths. In the tests made for this script such a run\n"
            "  gave the voxel the same result with the Mie and the Henyey-Greenstein phase function, i.e. the\n"
            "  cloud's Mie phase function (and so the bow) did not act, and the DoLP was several times too low.\n"
            "  Use one wavelength: BAND_SET = \"modis\" or \"rgb\" with a single CHANNEL (\"red\")."
        )
    theta_reference = np.arange(90.0, 180.0001, 0.25)
    reference = single_scattering(spectrum, voxel["reff_um"], theta_reference)
    print("  " + describe_bow(theta_reference, reference["dolp"][-1], reference["f11"][-1]) + f" ({labels[-1]})")
    if PLAN_ONLY:
        return

    folder = output_root / path.name.split(".")[0]
    # The longest file written for an angle has a name of about 150 characters; Windows refuses paths over
    # 260, and says only "no such file" after the (long) render.
    longest = len(str(folder / "theta_000.0")) + 1 + 150
    if sys.platform == "win32" and longest > 255:
        raise ValueError(
            f"the output paths would be about {longest} characters long, over Windows' limit of 260: "
            f"set OUTPUT_DIR (now {output_root}) to a shorter path"
        )
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "voxel.json").write_text(json.dumps({"dataset": path.name, "how": how, "cloud_file": str(voxel_file), **voxel}, indent=2))
    rows = []
    for number, (theta, sza, saa) in enumerate(plan, 1):
        print(f"--- angle {number}/{len(plan)}: scattering {theta:g} deg (sun zenith {sza:.2f}, azimuth {saa:.2f}) ---")
        started = time.time()
        playground_core.OUTPUT_DIR = folder / f"theta_{theta:05.1f}"
        simulation.render(dataclasses.replace(config, sza_deg=sza, saa_deg=saa), on_status=print, cloud=cloud)
        [stokes_file] = playground_core.OUTPUT_DIR.glob("*_stokes.npy")
        rows.append({"theta": theta, "sza": sza, "saa": saa, **polarization_of(np.load(stokes_file))})
        print("  DoLP " + ", ".join(f"{label}: {100 * d:.2f} +- {100 * e:.2f} %"
                                    for label, d, e in zip(labels, rows[-1]["dolp"], rows[-1]["dolp_error"]))
              + f"   ({time.time() - started:.0f} s)")
        write_outputs(folder, path.name.split(".")[0], labels, rows, reference, theta_reference, (vza, vaa), config, voxel)
    print(f"PDF: {folder / 'cloudbow_voxel.pdf'}")


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
    print(f"{len(datasets)} BOMEX file(s); {len(theta_list)} scattering angles ({theta_list[0]:g} to {theta_list[-1]:g} deg)")

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
