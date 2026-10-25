"""Estimate how many voxels one pixel sees, for BOMEX datasets - no GUI, no
display, no GPU. The same estimate as the playground's "Voxels per pixel"
button, for every BOMEX file in a folder and for several image resolutions at
once, printed in the terminal.

Every pixel of a render is one line of sight, on a regular grid over the cloud
field, so a pixel's footprint is the field's width divided by the pixel count.
When that footprint is much smaller than a voxel, neighbouring pixels see the
same voxels and a lower resolution gives the same information. For each
resolution and each view of the scenario (the single satellite's, or each
satellite of the formation) the script prints

  - the voxels in a pixel's footprint (pixel footprint / voxel projection;
    below 1: neighbouring pixels see the same voxels),
  - how many voxels a line of sight crosses in the cloud layers, and the share
    of them that its right and upper neighbours' lines of sight cross too,

and, at the end, the resolution at which a pixel is as large as a voxel.

To use it on another machine, edit only the SETTINGS block below - mainly
BOMEX_DIR, the folder that holds the BOMEX files - and run:

    python estimate_voxels_headless.py

The first run on a dataset builds its cloud file (about half a minute for the
full field), which run_bomex_headless.py then reuses, and the other way round.
"""

import dataclasses
import sys
import time
import traceback
from pathlib import Path

# ============================== SETTINGS ==============================
# Everything you normally change is here. A relative path is taken relative to
# this file's folder, not to the directory the script is launched from; an
# absolute path also works.

# Where the BOMEX netCDF files are: the folder, and which files in it. Files
# without the FNCD droplet spectrum (e.g. the "...com3D_int_1.nc" files) are
# skipped with a message. See run_bomex_headless.py for the same settings.
BOMEX_DIR = "/data/bomex"
BOMEX_PATTERN = "*.com3D_int_2.nc"
# Or name the files yourself (full paths); when not empty, BOMEX_DIR and
# BOMEX_PATTERN are ignored.
BOMEX_FILES: list[str] = []

# The saved scenario that gives the viewing geometry (saved from the GUI with
# "Save configuration"): the satellite or the formation, and whether the domain
# is periodic. Its cloud settings are ignored - the ones below replace them.
SCENARIO_CONFIG = "configs/scenario.json"

# Which image resolutions to estimate, by name (the GUI's "Image resolution"
# list). An empty list means the scenario's own resolution only.
RESOLUTIONS = [
    "Small (32x32)",
    "Medium (64x64)",
    "Large (128x128)",
    "XL (256x256)",
    "XXL (512x512)",
    "Huge (1024x1024)",
]
# Formation scenarios: which satellites to estimate (0-9), or None for the ones
# the scenario renders. Ignored for a single-satellite scenario.
SATELLITES: list[int] | None = None

# How the cloud's extinction coefficient is computed ("q2" or "mie") and the
# crop (0: the whole field) - these decide which cloud file is used. They do not
# change the voxel counts much, but use the same values as the renders.
EXTINCTION_METHOD = "q2"
CROP_CELLS = 0

# True: the full report for every view and resolution; False: one line each.
DETAILED = False

# Where the cloud files are kept (shared with run_bomex_headless.py).
GENERATED_CLOUD_DIR = "generated_clouds"
# =======================================================================

# The droplet phase function does not matter for this estimate.
PHASE_FUNCTION = "mie"

here = Path(__file__).parent


def resolve(path: str) -> Path:
    return (here / path).expanduser()  # an absolute path wins


# The no-display setup (Traits backend, SMARTG_DIR_AUXDATA, the import
# workaround for main.py) lives in run_from_config.py; importing it only does
# that setup and does not run anything.
import run_from_config  # noqa: E402

playground_core = run_from_config.playground_core


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


def report(views: list[dict]) -> None:
    """Print the views grouped by resolution, then the matched resolution."""
    for resolution in dict.fromkeys(v["resolution"] for v in views):
        print(f"\n--- image resolution {resolution} ---")
        for v in views:
            if v["resolution"] != resolution:
                continue
            args = (v["label"], v["stats"], v["vza_deg"], v["width_px"], v["height_px"])
            if DETAILED:
                print(playground_core.format_pixel_voxel_report(*args))
            else:
                print(playground_core.format_pixel_voxel_summary(*args))

    # The resolution at which a pixel is as large as a voxel does not depend on
    # the one used to compute it, so read it off the first resolution's views.
    first = views[0]["resolution"]
    matched = {v["label"]: v["stats"]["matched_width_px"] for v in views if v["resolution"] == first}
    print("\nPixels across at which a pixel is as large as a voxel:")
    for label, width in matched.items():
        print(f"  {label}: about {width:.0f}")
    low, high = min(matched.values()), max(matched.values())
    if len(matched) == 1 or round(low) == round(high):
        print(
            f"Finer than about {low:.0f} pixels across, neighbouring pixels start to see the same voxels."
        )
    else:
        print(
            f"Finer than about {low:.0f} pixels across, neighbouring pixels start to see the same voxels in "
            f"at least one view; finer than about {high:.0f}, in every view."
        )
    print("Coarser than that, a pixel covers several voxels but samples only one line of sight through them.")


def main() -> int:
    config_path = resolve(SCENARIO_CONFIG)
    try:
        base = playground_core.PlaygroundConfig.load(config_path)
    except (OSError, ValueError) as exc:
        print(f"Could not load {config_path}: {exc}", file=sys.stderr)
        return 2

    datasets = find_datasets()
    if not datasets:
        print(f"No BOMEX files found ({BOMEX_FILES or resolve(BOMEX_DIR) / BOMEX_PATTERN}).", file=sys.stderr)
        return 2

    playground_core.GENERATED_CLOUD_DIR = resolve(GENERATED_CLOUD_DIR)
    print(f"{len(datasets)} BOMEX file(s); scenario {config_path.name} ({base.mode}); "
          f"extinction {EXTINCTION_METHOD}, crop {CROP_CELLS or 'none'}")

    # The geometry is the same for every dataset: build it once.
    simulation = playground_core.PlaygroundSimulation.from_config(base)

    done, skipped, failed = [], [], []
    for number, path in enumerate(datasets, 1):
        print(f"\n===== [{number}/{len(datasets)}] {path.name} =====")
        started = time.time()
        try:
            if not path.is_file():
                raise FileNotFoundError(f"{path} does not exist")
            if not has_droplet_spectrum(path):
                print("No FNCD droplet spectrum in this file: skipped.")
                skipped.append(path.name)
                continue

            changes = dict(
                cloud_source=playground_core.CLOUD_BOMEX,
                bomex_netcdf=str(path),
                extinction_method=EXTINCTION_METHOD,
                bomex_crop_cells=CROP_CELLS,
                phase_function=PHASE_FUNCTION,
            )
            if SATELLITES is not None and base.mode == playground_core.FORMATION_MODE:
                changes["satellites_to_render"] = list(SATELLITES)
            config = dataclasses.replace(base, **changes)
            config.validate()

            views = simulation.pixel_voxel_views(config, on_status=print, resolutions=RESOLUTIONS or None)
            report(views)
            done.append(path.name)
            print(f"\nDone in {time.time() - started:.0f} s.")
        except Exception:  # one bad file should not stop the rest of the batch
            traceback.print_exc()
            failed.append(path.name)

    print(f"\nDone: {len(done)}, skipped: {len(skipped)}, failed: {len(failed)}")
    for name in failed:
        print(f"  failed: {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
