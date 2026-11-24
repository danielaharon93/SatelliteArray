"""Degree of linear polarization against scattering angle for ONE HOMOGENEOUS VOXEL - headlessly.

A simple case for finding the cloudbow before trying realistic clouds. It is run_cloudbow_headless.py
with a different cloud: instead of a whole BOMEX field, the cloud is a single voxel of it, so the
medium is homogeneous (one extinction coefficient, one droplet effective radius). The voxel is taken
from the BOMEX file: of the voxels whose optical depth (extinction x the voxel's thickness) is at
least MIN_VOXEL_OPTICAL_DEPTH - the cores of the clouds, not their hazy edges or the clear air - the
one whose effective radius is the median. It is written as a small cloud file (the layers below it
empty, as in the BOMEX cloud file, so the atmosphere is still modelled all the way down).

The voxel supplies only the PHYSICAL PROPERTIES of the medium (its extinction and droplet radius); the
medium's SIZE is MEDIUM_SIZE_M (x, y, z in metres; the voxel's own 20 m size by default). Being homogeneous,
a bigger medium only changes its optical depth (the extinction x its thickness) and its horizontal extent.

With PERIODIC the medium repeats horizontally, which makes it an infinite homogeneous layer as thick as
the medium (20 m by default): every pixel of the image then sees the same medium, so the image's mean Stokes
parameters are the right quantity (no sea or cloud-edge pixels to separate) and the pixels only add
photons. (SMART-G's periodic grid needs at least 2 cells per direction, so the file holds a 2 x 2
block of identical voxels.) Without PERIODIC, the voxel is one isolated 20 m cube, which is mostly
transparent to a camera that looks at the sea behind it.

Unlike run_cloudbow_headless.py, the SUN is fixed - the scenario's sun_zenith and sun_azimuth - and the
scattering angle is changed by moving the CAMERA, in the vertical plane of the sun. With the signed view
zenith t = sun zenith - (180 deg - scattering angle), the camera is on the sun's side (the sun's azimuth,
view zenith t) for t >= 0 and on the opposite side (azimuth + 180 deg, view zenith -t) for t < 0, so that

    scattering angle = 180 deg - |sun zenith - t|    (the view zenith is at the cloud's height)

the camera crossing the nadir at scattering angle 180 deg - sun zenith. The satellite is the scenario's (its
position follows from the TLE and the time), and each camera direction is made by pointing it: the script
finds the off-nadir angle and the pointing azimuth that give the wanted view direction at the cloud (they
replace the scenario's own). A camera direction can be SHADOWED BY THE EARTH: if the camera would be at or
below the horizon of the cloud (view zenith 90 deg or more), or the satellite cannot point at it (past the
Earth's limb, or beyond the playground's largest off-nadir angle), the script prints a warning and skips
that angle.

Everything else - the satellite, the sun, the surface, the atmosphere, the resolution, the photons, the
extinction method and the phase function - is the scenario's own (configs/scenario.json). The exception
is the wavelength: BAND_SET (and WAVELENGTH_NM) below select ONE wavelength (see there for why).

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
# The size of the medium, (x, y, z) in metres. The voxel supplies only the PHYSICAL PROPERTIES of the
# medium (the extinction per km and the droplet effective radius); the medium is homogeneous, so its size
# only sets its optical depth (the extinction x z) and its horizontal extent (the period of a periodic
# layer, or the width of an isolated block). It is centred, vertically, on the voxel's height. None: the
# voxel's own size (20 x 20 m horizontally and its thickness, 20 m for the BOMEX layers; a periodic layer
# is then a 2 x 2 block of them, 40 m wide).
MEDIUM_SIZE_M: tuple[float, float, float] | None = None

# The scattering angles, in degrees (at most 180): from START to STOP (included) every STEP. The sun is the
# scenario's, so these are made by moving the camera. The primary bow of droplets like these is a few
# degrees wide, near 140 degrees, so a step of 1 or less is needed.
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
# this script uses one wavelength by default: the "single" band set, at the scenario's wavelength_nm (or
# WAVELENGTH_NM here). A run with several wavelengths prints a warning. None keeps the scenario's value.
# (Other one-wavelength choices: BAND_SET "modis" or "rgb" with a single CHANNEL.) GRID_STEP_NM is for the
# grid band sets.
BAND_SET: str | None = "single"
WAVELENGTH_NM: float | None = None
CHANNEL: str | None = None
GRID_STEP_NM: float | None = None

# True: only choose the voxel, write its file and print the plan and the single-scattering bow (no
# render; needs no GPU).
PLAN_ONLY = False

# One folder per dataset, in it one folder per angle, and cloudbow_voxel.pdf, cloudbow_voxel.csv and
# voxel.json (which voxel it was).
OUTPUT_DIR = "cloudbow_voxel_runs"
# =======================================================================

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


def angles() -> list[float]:
    if ANGLE_STEP_DEG <= 0 or ANGLE_STOP_DEG < ANGLE_START_DEG or ANGLE_STOP_DEG > 180.0:
        sys.exit("Set ANGLE_START_DEG <= ANGLE_STOP_DEG <= 180 and ANGLE_STEP_DEG > 0.")
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


def medium_of(field, voxel: dict, cells: int, size_m: tuple[float, float, float] | None) -> dict:
    """The voxel's properties with the size of the medium: `size_m` (x, y, z) in metres, or the voxel's own
    size when None (`cells` of its horizontal size across, and its thickness). The medium is centred,
    vertically, on the voxel, so the aim point of the view does not move. Adds `size_m`, the medium's
    `z_bottom_km`, `z_top_km`, `thickness_km` and `optical_depth` (extinction x thickness)."""
    if size_m is None:
        dx = float(np.diff(field["x_bounds"].values)[0]) * 1000.0
        dy = float(np.diff(field["y_bounds"].values)[0]) * 1000.0
        size_m = (cells * dx, cells * dy, 1000.0 * voxel["thickness_km"])
    if len(size_m) != 3 or not all(v > 0 for v in size_m):
        raise ValueError(f"MEDIUM_SIZE_M must be three positive numbers (x, y, z) in metres, not {size_m!r}")
    z_centre = 0.5 * (voxel["z_bottom_km"] + voxel["z_top_km"])
    z_bottom, z_top = z_centre - size_m[2] / 2000.0, z_centre + size_m[2] / 2000.0
    if z_bottom < 0.0:
        raise ValueError(
            f"a medium {size_m[2]:g} m thick, centred on the voxel at {1000 * z_centre:.0f} m, would reach below the "
            f"ground: use a thickness of at most {2000 * z_centre:.0f} m, or another voxel"
        )
    thickness = z_top - z_bottom
    return {**voxel, "size_m": [float(v) for v in size_m], "z_bottom_km": z_bottom, "z_top_km": z_top,
            "thickness_km": thickness, "optical_depth": voxel["extinction_per_km"] * thickness}


def write_voxel_cloud(field, info: dict, source: str, path: Path, cells: int) -> Path:
    """The cumulus-format cloud file of the medium `info` (see medium_of): `cells` x `cells` identical cells
    across its x, y size (one, or a block that a periodic domain repeats), one layer of its thickness, the
    BOMEX field's own (empty) layers below it from the ground up, and above it the empty padding the BOMEX
    file has (the atmosphere is modelled only on the grid's layers)."""
    import cumulus_file

    bottom, top = info["z_bottom_km"], info["z_top_km"]
    native = field["z_bounds"].values
    below = native[native < bottom - 1e-9]  # the BOMEX layers under the medium (the ground, 0 km, is the first)
    z_bounds = np.concatenate([below, [bottom, top]])
    nz_data = len(z_bounds) - 1
    pad = [e for e in cumulus_file.TOP_PAD_EDGES_KM if e > top + cumulus_file.TOP_PAD_MIN_KM]
    dx, dy = info["size_m"][0] / 1000.0 / cells, info["size_m"][1] / 1000.0 / cells
    grid = cumulus_file.CumulusGrid(cells, cells, dx, dy, np.concatenate([z_bounds, pad]), nz_data=nz_data)
    cf = cumulus_file.create_cumulus_file(path, grid, f"{cells}x{cells} identical cells, {info['size_m']} m, from {source}: {info}")
    ext = np.zeros((nz_data, cells, cells))
    reff = np.zeros((nz_data, cells, cells))
    ext[-1], reff[-1] = info["extinction_per_km"], info["reff_um"]
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
                  theta_reference: np.ndarray, sun: tuple, config, voxel: dict) -> None:
    """cloudbow_voxel.csv (one line per angle and channel) and cloudbow_voxel.pdf."""
    with (folder / "cloudbow_voxel.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["scattering_angle_deg", "view_zenith_deg", "view_azimuth_deg", "off_nadir_deg", "pointing_azimuth_deg",
             "sun_zenith_deg", "sun_azimuth_deg", "channel", "I", "Q", "U", "V",
             "dolp_percent", "dolp_error_percent", "aolp_deg", "single_scattering_dolp_percent", "single_scattering_F11"]
        )
        for row in rows:
            for k, label in enumerate(labels):
                s = row["stokes_mean"][:, k]
                dolp_ss = 100 * float(np.interp(row["theta"], theta_reference, reference["dolp"][k]))
                f11_ss = float(np.interp(row["theta"], theta_reference, reference["f11"][k]))
                writer.writerow(
                    [f"{row['theta']:.3f}", f"{row['vza']:.3f}", f"{row['vaa']:.3f}", f"{row['eta']:.3f}", f"{row['phi']:.3f}",
                     f"{sun[0]:.3f}", f"{sun[1]:.3f}", label,
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
    sza, saa = sun
    figure.suptitle("Cloudbow of one homogeneous BOMEX voxel: polarization and reflectance against scattering angle", fontsize=10)
    top.set_title(
        f"{dataset}\n"
        f"medium: {voxel['size_m'][0]:g} x {voxel['size_m'][1]:g} x {voxel['size_m'][2]:g} m, extinction "
        f"{voxel['extinction_per_km']:.1f} /km, optical depth {voxel['optical_depth']:.2f}, "
        f"droplet radius {voxel['reff_um']:.2f} um, {'periodic layer' if config.periodic else 'isolated block'}\n"
        f"sun zenith {sza:.1f} deg, azimuth {saa:.1f} deg (the scenario's); camera in the sun's vertical plane, "
        f"view zenith {min(r['vza'] for r in rows):.1f} to {max(r['vza'] for r in rows):.1f} deg; "
        f"{config.band_set}, {config.resolution}, {config.photons_per_pixel:g} photons/pixel",
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
    # almost no signal), so the periodic layer is a 2 x 2 block of identical cells: the same medium.
    cells = PERIODIC_CELLS if config.periodic else 1
    voxel = medium_of(field, voxel, cells, MEDIUM_SIZE_M)  # from here on: the medium (the voxel's properties, its size)
    x_m, y_m, z_m = voxel["size_m"]
    print(f"Medium: {x_m:g} x {y_m:g} x {z_m:g} m ({'repeated horizontally' if config.periodic else 'isolated'}), height "
          f"{voxel['z_bottom_km']:.3f}-{voxel['z_top_km']:.3f} km, optical depth {voxel['optical_depth']:.2f} "
          f"(extinction x {z_m:g} m)")
    key = hashlib.sha1(
        f"{field_file}|{iz},{iy},{ix}|{voxel['extinction_per_km']}|{voxel['reff_um']}|{cells}|{voxel['size_m']}|v3".encode()
    ).hexdigest()[:10]
    voxel_file = resolve(GENERATED_CLOUD_DIR) / f"voxel_{key}.dat"
    if not voxel_file.exists():
        voxel_file.parent.mkdir(parents=True, exist_ok=True)
        partial = voxel_file.with_name(voxel_file.name + ".partial")
        write_voxel_cloud(field, voxel, path.name, partial, cells)
        partial.replace(voxel_file)
    cloud = playground_core.CloudSource(voxel_file, config.phase_function, config.hg_g)
    print(f"Cloud (the voxel, from the scenario's physics): extinction {config.extinction_method}, phase function "
          f"{config.phase_function}, {'periodic layer' if config.periodic else 'isolated block'}; file {voxel_file.name}")

    simulation = playground_core.PlaygroundSimulation.from_config(config)
    sza, saa = config.sza_deg, config.saa_deg
    print(f"Sun (the scenario's): zenith {sza:.2f} deg, azimuth {saa:.2f} deg; satellite at {simulation.time_utc}")

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
    print(f"  {len(plan)} scattering angles {plan[0]['theta']:g}-{plan[-1]['theta']:g} deg: view zenith "
          f"{min(p['vza'] for p in plan):.1f} to {max(p['vza'] for p in plan):.1f} deg")

    spectrum = playground_core.config_spectrum(config)
    labels = tuple(spectrum.labels)
    most = max(len(part.wavelengths_nm) for part in spectrum.runs)
    if most > 1:
        print(
            f"  WARNING: a SMART-G run here has {most} wavelengths. In the tests made for this script such a run\n"
            "  gave the voxel the same result with the Mie and the Henyey-Greenstein phase function, i.e. the\n"
            "  cloud's Mie phase function (and so the bow) did not act, and the DoLP was several times too low.\n"
            "  Use one wavelength: BAND_SET = \"single\" (the scenario's wavelength_nm)."
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
    for number, step in enumerate(plan, 1):
        theta = step["theta"]
        print(f"--- angle {number}/{len(plan)}: scattering {theta:g} deg (view zenith {step['vza']:.2f}, azimuth "
              f"{step['vaa']:.2f}) ---")
        started = time.time()
        playground_core.OUTPUT_DIR = folder / f"theta_{theta:05.1f}"
        simulation.render(
            dataclasses.replace(config, off_nadir_deg=step["eta"], view_azimuth_deg=step["phi"]), on_status=print, cloud=cloud
        )
        [stokes_file] = playground_core.OUTPUT_DIR.glob("*_stokes.npy")
        rows.append({**step, **polarization_of(np.load(stokes_file))})
        print("  DoLP " + ", ".join(f"{label}: {100 * d:.2f} +- {100 * e:.2f} %"
                                    for label, d, e in zip(labels, rows[-1]["dolp"], rows[-1]["dolp_error"]))
              + f"   ({time.time() - started:.0f} s)")
        write_outputs(folder, path.name.split(".")[0], labels, rows, reference, theta_reference, (sza, saa), config, voxel)
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
