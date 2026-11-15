"""Render a bare sea with no atmosphere and no cloud, headlessly.

The scenario (satellite geometry, sun, wind speed, resolution, photons, band set,
sea water body...) comes from a .json file saved by smartg_playground.py ("Save
configuration"). This script forces the two things that make it a bare-sea scene:

  * no cloud,
  * no atmosphere at all (no air molecules, gases or aerosol): the sun lights the
    sea directly and the sensor sees it without haze or attenuation.

The surface is the ocean. The sea's water body (water_body, water_model,
chlorophyll_mg_m3, sea_spectrum_file, matte_offset) is whatever the file says,
unless you override it below. The aerosol, ozone, water vapour and pressure settings
of the file are ignored (there is no atmosphere). One render, one set of images.

Needs a CUDA GPU and SMART-G's auxiliary data (see run_from_config.py). Run:

    python run_ocean_no_atmosphere_headless.py
"""

import dataclasses
import sys
from pathlib import Path

# ============================== SETTINGS ==============================
# A relative path is taken relative to this file's folder; an absolute path works too.

# The scenario saved from the GUI.
SCENARIO_CONFIG = "configs/scenario.json"

# Override the file's sea water body for this run, or None to keep the file's:
# WATER_BODY True/False (False: the sea only reflects, no light from the water),
# CHLOROPHYLL_MG_M3 the concentration (water_model "chlorophyll" only).
WATER_BODY: bool | None = None
CHLOROPHYLL_MG_M3: float | None = None

# Override the file's spectral settings, or None to keep them:
# BAND_SET "rgb" / "modis" / "grid", CHANNEL "all" / "blue" / "green" / "red".
BAND_SET: str | None = None
CHANNEL: str | None = None

# True: besides the reflectance (<name>.npy), also save the radiance in
# W m-2 um-1 sr-1 (<name>_radiance.npy and .json), as run_bomex_headless.py does.
SAVE_RADIANCE = True

# True: also save wavelength_distribution.pdf, the probability with which each photon's
# wavelength is drawn, in the output folder.
SAVE_WAVELENGTH_DISTRIBUTION = True

# Where the images are written.
OUTPUT_DIR = "ocean_no_atmosphere_runs"
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
        # require_dataset=False: the file may name a cloud dataset, which this scene does not use
        base = playground_core.PlaygroundConfig.load(config_path, require_dataset=False)
    except (OSError, ValueError) as exc:
        print(f"Could not load {config_path}: {exc}", file=sys.stderr)
        return 2

    overrides = {
        name: value
        for name, value in (
            ("water_body", WATER_BODY),
            ("chlorophyll_mg_m3", CHLOROPHYLL_MG_M3),
            ("band_set", BAND_SET),
            ("channel", CHANNEL),
        )
        if value is not None
    }
    try:
        config = dataclasses.replace(
            base,
            surface=playground_core.SURFACE_OCEAN,
            cloud_source=playground_core.CLOUD_NONE,
            no_atmosphere=True,
            **overrides,
        )
        config.validate()
    except ValueError as exc:
        print(f"Bad settings: {exc}", file=sys.stderr)
        return 2

    playground_core.OUTPUT_DIR = resolve(OUTPUT_DIR)
    playground_core.SAVE_RADIANCE = SAVE_RADIANCE
    water = f"{config.water_model}, chlorophyll {config.chlorophyll_mg_m3:g} mg/m3" if config.water_body else "reflection only"
    print(f"Bare sea, no atmosphere, no cloud; sea water body: {water}; output {playground_core.OUTPUT_DIR}")

    playground_core.run_config(config)
    if SAVE_WAVELENGTH_DISTRIBUTION:
        pdf = playground_core.OUTPUT_DIR / "wavelength_distribution.pdf"
        playground_core.save_wavelength_distribution_figure(
            playground_core.config_spectrum(config), pdf, title=f"{config.band_set} band set"
        )
        print(f"Wavelength distribution: {pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
