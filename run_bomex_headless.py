"""Render BOMEX clouds with SMART-G, headlessly - no GUI, no display.

For each BOMEX netCDF file found on disk (the `...com3D_int_2.nc` files, which
hold the droplet spectrum `FNCD`), this script

  1. builds a cumulus.dat-format cloud file from it (extinction and effective
     radius per voxel - see cloud_microphysics.py and cumulus_file.py), and
  2. renders the scenario of a saved config on that cloud with SMART-G
     (same as run_from_config.py, but with the BOMEX cloud).

The droplet phase function is always the Mie one: SMART-G takes it from its
water-cloud table (clouds/wc_sol.nc), by each voxel's effective radius.

To use it on another machine, edit only the SETTINGS block below - mainly
BOMEX_DIR, the folder that holds the BOMEX files - and run:

    python run_bomex_headless.py

Needs a CUDA GPU and SMART-G's auxiliary data (SMARTG_DIR_AUXDATA, default: the
smartg_auxdata/ folder next to this file) unless RENDER is False, which only
builds the cloud files and needs neither.
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

# Where the BOMEX netCDF files are: the folder, and which files in it. The
# pattern is searched in the folder only; use "**/*.com3D_int_2.nc" to include
# its subfolders. Files without the FNCD droplet spectrum (e.g. the
# "...com3D_int_1.nc" files) are skipped with a message.
BOMEX_DIR = "/data/bomex"
BOMEX_PATTERN = "*.com3D_int_2.nc"
# Or name the files yourself (full paths); when not empty, BOMEX_DIR and
# BOMEX_PATTERN are ignored.
BOMEX_FILES: list[str] = []

# The saved scenario to render on each cloud (saved from the GUI with "Save
# configuration"): satellite geometry, sun, surface, resolution, photons...
# Its cloud settings are ignored - the settings below replace them.
SCENARIO_CONFIG = "configs/scenario.json"

# How the cloud's extinction coefficient is computed from the droplet spectrum:
# "q2" (the paper's approximation Q = 2) or "mie" (the Mie Q of each radius bin).
EXTINCTION_METHOD = "q2"
# 0 keeps each whole field; otherwise a square of this many cells (at least 16)
# around the column with the largest optical depth.
CROP_CELLS = 0

# False: only build the cloud files (no GPU, no SMART-G needed).
RENDER = True

# Where things are written: the cloud files (one per dataset and setting, reused
# on later runs), and the images (one subfolder per dataset).
GENERATED_CLOUD_DIR = "generated_clouds"
OUTPUT_DIR = "bomex_runs"
# =======================================================================

# The droplet phase function. Fixed: SMART-G's Mie phase functions (see above).
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
    output_root = resolve(OUTPUT_DIR)
    print(f"{len(datasets)} BOMEX file(s); extinction {EXTINCTION_METHOD}, "
          f"crop {CROP_CELLS or 'none'}, phase {PHASE_FUNCTION}, render {RENDER}")

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

            config = dataclasses.replace(
                base,
                cloud_source=playground_core.CLOUD_BOMEX,
                bomex_netcdf=str(path),
                extinction_method=EXTINCTION_METHOD,
                bomex_crop_cells=CROP_CELLS,
                phase_function=PHASE_FUNCTION,
            )
            config.validate()
            if RENDER:
                playground_core.OUTPUT_DIR = output_root / path.name.split(".")[0]
                playground_core.run_config(config)
                print(f"Images in {playground_core.OUTPUT_DIR}")
            else:
                cloud_file = playground_core.bomex_cloud_file(
                    path, EXTINCTION_METHOD, CROP_CELLS, on_status=print
                )
                print(f"Cloud file: {cloud_file}")
            done.append(path.name)
            print(f"Done in {time.time() - started:.0f} s.")
        except Exception:  # one bad file should not stop the rest of the batch
            traceback.print_exc()
            failed.append(path.name)

    print(f"\nDone: {len(done)}, skipped: {len(skipped)}, failed: {len(failed)}")
    for name in failed:
        print(f"  failed: {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
