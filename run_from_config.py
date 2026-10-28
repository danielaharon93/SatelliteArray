"""Run a saved SMART-G playground scenario without any GUI or display.

Set a scenario up in smartg_playground.py ("Save configuration..."), copy
the .json file to a machine with a CUDA GPU (e.g. a remote server with no
desktop), set CONFIG_PATH below to it, and run this file:

    python run_from_config.py

The images are written to smartg_playground_images/ next to this file, named
as the GUI names them. The config pins down the satellite geometry (time and
TLE), so the render matches what the GUI would have produced.
"""

import dataclasses
import os
import sys
from pathlib import Path

# The scenario file to render. A relative path is taken relative to this
# file's folder, not the directory the script happens to be launched from.
CONFIG_PATH = "configs/scenario.json"

# Which wavelengths to simulate. None uses the scenario file's own "channel"
# setting (set in the GUI's "Color channel"); or override it here for this run:
# "all" (blue, green and red: 440, 550 and 670 nm) or one of "blue", "green",
# "red" (a single wavelength, written as a grayscale image).
CHANNEL = None

# What a channel is: "rgb" (single wavelengths: 440, 550 and 670 nm), "modis"
# (the MODIS Terra channels b03, b04 and b01, simulated as REPTRAN bands) or
# "grid" (a grid of wavelengths over the camera's range, see grid_range_nm in the
# scenario file). None uses the scenario file's own "band_set" setting.
BAND_SET = None

# No display on this kind of machine: keep Traits/Pyface on their no-GUI
# backend instead of trying to start Qt. Must be set before they're imported.
os.environ.setdefault("ETS_TOOLKIT", "null")

# smartg reads its auxiliary data location from this variable at import time
# and raises NameError if it's unset. Default to the folder next to this file
# (where main.ensure_smartg_auxdata() downloads it); an existing value wins.
os.environ.setdefault(
    "SMARTG_DIR_AUXDATA", str(Path(__file__).parent / "smartg_auxdata")
)

try:
    import playground_core
except NotImplementedError:
    # main.py builds its (here unused) Mayavi window's View at import time,
    # which the no-GUI backend can't do for the 3D scene editor. Swap in a
    # placeholder editor for this process, then import again - nothing here
    # ever opens a window.
    import mayavi.core.ui.api as _mayavi_ui

    _mayavi_ui.SceneEditor = lambda *args, **kwargs: None
    for _name in [m for m in sys.modules if m in ("main", "playground_core")]:
        del sys.modules[_name]
    import playground_core


def main() -> int:
    config_path = Path(__file__).parent / CONFIG_PATH  # absolute CONFIG_PATH wins
    try:
        config = playground_core.PlaygroundConfig.load(config_path)
    except (OSError, ValueError) as exc:
        print(f"Could not load {config_path}: {exc}", file=sys.stderr)
        return 2

    overrides = {name: value for name, value in (("channel", CHANNEL), ("band_set", BAND_SET)) if value is not None}
    if overrides:
        config = dataclasses.replace(config, **overrides)
        try:
            config.validate()
        except ValueError as exc:
            print(f"Bad CHANNEL / BAND_SET setting in {Path(__file__).name}: {exc}", file=sys.stderr)
            return 2

    playground_core.run_config(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
