"""Render a cloud-free scene with the sea's chlorophyll water body, headlessly.

The scenario (satellite geometry, sun, sea, aerosol, resolution, photons, band
set...) comes from a .json file saved by smartg_playground.py ("Save
configuration"). This script forces the parts that make it a chlorophyll scene:

  * no cloud,
  * the ocean surface with a water body, represented as a column of pure water
    and chlorophyll-bearing particles (water_model "chlorophyll"), so the sea's
    colour follows from the chlorophyll concentration.

Everything else is taken from the file. List several chlorophyll values below to
get one render per value (e.g. to compare a blue and a green sea). Needs a CUDA
GPU and SMART-G's auxiliary data (see run_from_config.py). Run:

    python run_chlorophyll_headless.py
"""

import dataclasses
import sys
import time
import traceback
from pathlib import Path

# ============================== SETTINGS ==============================
# A relative path is taken relative to this file's folder; an absolute path works too.

# The scenario saved from the GUI. Its cloud settings and sea-colour model are
# replaced as described above.
SCENARIO_CONFIG = "configs/scenario.json"

# Chlorophyll concentrations to simulate, in mg/m3 (one render each). Typical: 0.1 open
# ocean (deep blue), 0.5 blue-teal, 1 and 2 blue-green to green, 5 turbid green-brown.
# None (or an empty list) uses the scenario file's own value.
CHLOROPHYLL_MG_M3: list[float] | None = [0.1, 0.5, 1.0, 2.0, 5.0]

# Override the file's spectral settings for this run, or None to keep them:
# BAND_SET "rgb" / "modis" / "grid", CHANNEL "all" / "blue" / "green" / "red".
BAND_SET: str | None = None
CHANNEL: str | None = None

# True: no atmosphere at all (no air molecules, gases or aerosol): the sea is lit by the sun
# directly and seen without haze, a reference for how much the atmosphere changes the image.
# None keeps the scenario file's own "no_atmosphere" setting.
NO_ATMOSPHERE: bool | None = None

# True: besides each image's reflectance (<name>.npy), also save its radiance in
# W m-2 um-1 sr-1 (<name>_radiance.npy and .json), as run_bomex_headless.py does.
SAVE_RADIANCE = True

# True: also save wavelength_distribution.pdf, the probability with which each photon's
# wavelength is drawn, in the output folder.
SAVE_WAVELENGTH_DISTRIBUTION = True

# Where the images are written (one subfolder per formation run, as in the GUI).
OUTPUT_DIR = "chlorophyll_runs"
# =======================================================================

here = Path(__file__).parent

# Importing run_from_config only does the no-display setup (Traits backend,
# SMARTG_DIR_AUXDATA, the main.py import workaround); it does not run anything.
import run_from_config  # noqa: E402

playground_core = run_from_config.playground_core


def resolve(path: str) -> Path:
    return (here / path).expanduser()  # an absolute path wins


def main() -> int:
    config_path = resolve(SCENARIO_CONFIG)
    try:
        base = playground_core.PlaygroundConfig.load(config_path, require_dataset=False)
    except (OSError, ValueError) as exc:
        print(f"Could not load {config_path}: {exc}", file=sys.stderr)
        return 2

    overrides = {
        name: value
        for name, value in (("band_set", BAND_SET), ("channel", CHANNEL), ("no_atmosphere", NO_ATMOSPHERE))
        if value is not None
    }
    values = list(CHLOROPHYLL_MG_M3 or [base.chlorophyll_mg_m3])

    playground_core.OUTPUT_DIR = resolve(OUTPUT_DIR)
    playground_core.SAVE_RADIANCE = SAVE_RADIANCE
    print(f"Chlorophyll values (mg/m3): {', '.join(f'{v:g}' for v in values)}; output {playground_core.OUTPUT_DIR}")

    done, failed = [], []
    for number, chl in enumerate(values, 1):
        print(f"\n===== [{number}/{len(values)}] chlorophyll {chl:g} mg/m3 =====")
        started = time.time()
        try:
            config = dataclasses.replace(
                base,
                surface=playground_core.SURFACE_OCEAN,
                water_body=True,
                water_model=playground_core.WATER_CHLOROPHYLL,
                chlorophyll_mg_m3=chl,
                cloud_source=playground_core.CLOUD_NONE,
                **overrides,
            )
            config.validate()
            playground_core.run_config(config)
            if SAVE_WAVELENGTH_DISTRIBUTION and number == 1:  # the same for every chlorophyll value
                pdf = playground_core.OUTPUT_DIR / "wavelength_distribution.pdf"
                playground_core.save_wavelength_distribution_figure(
                    playground_core.config_spectrum(config), pdf, title=f"{config.band_set} band set"
                )
                print(f"Wavelength distribution: {pdf}")
            done.append(chl)
            print(f"Done in {time.time() - started:.0f} s.")
        except Exception:  # one bad value should not stop the rest
            traceback.print_exc()
            failed.append(chl)

    print(f"\nDone: {len(done)}, failed: {len(failed)}")
    for chl in failed:
        print(f"  failed: chlorophyll {chl:g}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
