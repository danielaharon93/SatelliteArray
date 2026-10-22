# Headless SMART-G pipeline

Set up a simulation scenario in the interactive GUI on a desktop machine, save it
to a file, then render it on a machine with a GPU but no display (e.g. a remote
Linux server).

```
 desktop (GUI)                               server (GPU, no display)
 ┌───────────────────────┐   git / copy     ┌──────────────────────────┐
 │ smartg_playground.py  │ ───────────────▶ │ run_from_config.py       │
 │  choose scenario      │  configs/*.json  │  renders the scenario    │
 │  "Save configuration" │                  │  → smartg_playground_    │
 └───────────────────────┘                  │    images/*.png          │
                                            └──────────────────────────┘
```

## Files

| File | Role |
|---|---|
| `smartg_playground.py` | Interactive GUI (Mayavi/Traits). Needs a display. Saves/loads scenarios. |
| `run_from_config.py` | Headless runner. No GUI, no display. Renders one saved scenario. |
| `playground_core.py` | Display-free core shared by both: the SMART-G render functions, the config format (`PlaygroundConfig`) and the simulation object (`PlaygroundSimulation`). |
| `pointing_noise.py` | Camera pointing-error model (see [Pointing error](#pointing-error)). |
| `configs/` | Saved scenarios (`*.json`). Commit these to move them to the server. |
| `smartg_playground_images/` | Output images (git-ignored). |

Because the GUI and the headless runner both call `PlaygroundSimulation.render()`
with a `PlaygroundConfig`, a scenario renders the same way in both.

## Workflow

### 1. Build a scenario in the GUI (desktop)

```
python smartg_playground.py
```

Set everything as wanted: sandbox mode, off-nadir/azimuth or satellites to render,
sun angles, periodic domain, resolution, photons per pixel, the pointing-error
controls, and **Renders in a headless run**. Use **Render** to try it out locally.

Click **Save configuration…** (row "Scenario file"). It opens in `configs/` and
defaults to `scenario.json`.

To reopen a saved scenario later, with its satellite geometry too:

```
python smartg_playground.py configs/scenario.json
```

The **Load configuration…** button inside a running window applies the *settings*
only; see [Geometry is pinned](#geometry-is-pinned).

### 2. Move it to the server

```
git add configs/scenario.json
git commit -m "Add scenario"
git push
```

then `git pull` on the server (or copy the file by any other means).

### 3. Run it headlessly (server)

In [run_from_config.py](run_from_config.py), set the constant near the top:

```python
CONFIG_PATH = "configs/scenario.json"   # relative to run_from_config.py's folder
```

and run:

```
python run_from_config.py
```

A relative `CONFIG_PATH` is resolved against the script's folder, not the working
directory; an absolute path also works. Progress and a per-render summary are
printed; images are written to `smartg_playground_images/` next to the script.
Exit code 2 means the config could not be loaded (the message says why).

Copy the images back from the server's `smartg_playground_images/` folder.

## The config file

JSON, written by the GUI. Every setting of the scenario plus the geometry moment:

| Key | Meaning |
|---|---|
| `version` | Format version (currently `1`). A different version is rejected. |
| `tle` | The TLE text (name line + two element lines) the orbit comes from. |
| `time_utc` | ISO 8601 UTC moment all satellite positions are computed at. |
| `mode` | `"Single satellite"` or `"Formation (frozen)"`. |
| `off_nadir_deg`, `view_azimuth_deg` | Single mode: viewing ray from the satellite (off-nadir 0–55°). |
| `satellites_to_render` | Formation mode: list of satellite indices 0–9. |
| `sza_deg`, `saa_deg` | Sun zenith / azimuth at the target (azimuth counterclockwise from local east). |
| `periodic` | Periodic vs. non-periodic (extended) cloud-field domain. |
| `surface` | `"lambertian"` (default: flat dark ground, albedo 0.05) or `"ocean"` (Cox-Munk wind-roughened sea: sun glint and Fresnel reflection, reflection only - no water-leaving radiance). Applies to the whole domain. |
| `wind_speed_ms` | Wind speed in m/s (0-30, default 5) for the `"ocean"` surface; ignored for `"lambertian"`. |
| `display_gain` | Brightness gain of the saved PNGs, `(gain × reflectance)^(1/1.8)` clipped to 1 (default 1.0: reflectance as it is, which keeps bright cloud pixels from saturating; the sea is dark). Only affects the PNG: the `.npy` next to it holds the raw reflectance. Raise it to brighten dark scenes. |
| `resolution` | `"Tiny (16x16)"`, `"Small (32x32)"` or `"Medium (64x64)"`. |
| `photons_per_pixel` | Photons per pixel; total sent to SMART-G = this × pixel count. |
| `cloud_source` | `"bundled"` (default: SMART-G's `cumulus.dat`) or `"bomex"` (a cloud built from a BOMEX droplet-spectrum dataset, see [The BOMEX cloud](#the-bomex-cloud)). |
| `bomex_netcdf` | Path to the BOMEX `…com3D_int_2.nc` file (the one with the `FNCD` droplet spectrum). Required for `"bomex"`. |
| `extinction_method` | `"q2"` (default: extinction with Q = 2, the paper's approximation) or `"mie"` (the Mie Q of each radius bin). Only used by `"bomex"`. |
| `bomex_crop_cells` | `0` (default: the whole field) or a square of this many cells (at least 16) around the thickest column. Only used by `"bomex"`. |
| `phase_function` | Droplet phase function, either cloud source: `"mie"` (default: the water-cloud table of SMART-G) or `"hg"` (Henyey-Greenstein). |
| `hg_g` | Asymmetry parameter for `"hg"`, in (-1, 1) (default 0.85, typical of water clouds). |
| `pointing_noise_enabled` | Turn the pointing-error model on. |
| `sigma_bias_deg`, `sigma_acquisition_deg`, `sigma_drift_deg`, `drift_tau_s`, `sigma_jitter_deg`, `yaw_factor`, `acquisition_interval_s` | Pointing-error parameters (below). |
| `noise_seed` | Random seed of the Monte-Carlo run. |
| `renders` | Headless only: number of acquisitions to render. |

The file can be edited by hand. On load it is validated: unknown keys, an unknown
mode/resolution, out-of-range angles, bad satellite indices, `renders < 1`, a
wrong `version` and an unparseable `time_utc` are all rejected with a message.

### The BOMEX cloud

With `"cloud_source": "bomex"`, the cloud is made from the droplet spectra in the netCDF
file (`cloud_microphysics.py`): per voxel, the extinction coefficient and the effective
radius. They are written once to a `cumulus.dat`-format file in `generated_clouds/`
(`cumulus_file.py`, git-ignored) and reused while the dataset and the extinction method
and crop stay the same; the first run takes about half a minute for the full 512 × 512
field. The file name carries the dataset's name, the settings and a hash, and line 1 of
the file records how it was made. The phase function and `g` are not in that file: they
are applied when the scene is built.

The dataset's path is stored in the config as it was typed, so on a server give a path that
exists there. Install `netCDF4` (in `requirements.txt`) on the server too. Output images get
a suffix naming the cloud and phase function (for example `_bomex_q2_crop200_hg0.85`); the
original settings keep their names.

#### Rendering many BOMEX files: `run_bomex_headless.py`

For a server with a folder of BOMEX files, `run_bomex_headless.py` does the whole job
with no GUI. Edit the SETTINGS block at the top (mainly `BOMEX_DIR`, the folder that
holds the files) and run `python run_bomex_headless.py`. For each file with an `FNCD`
spectrum it builds the cloud file and renders the scenario of `SCENARIO_CONFIG` on it,
writing the images to `bomex_runs/<dataset name>/`. Files without `FNCD` are skipped, a
failing file does not stop the others, and the exit code is 1 if any failed. The phase
function is always SMART-G's Mie one. With `RENDER = False` it only builds the cloud
files, which needs no GPU.

### Geometry is pinned

The GUI normally places satellites at "now" from a live TLE. That would make a later
headless run render a *different* geometry, so the config stores `tle` and `time_utc`
and the headless run rebuilds exactly that geometry. It needs no TLE or ephemeris
download. The sun is set directly by `sza_deg`/`saa_deg`, not from an ephemeris.

Consequence: to re-run a scenario at a new moment, open the GUI fresh (no config
argument) and save again.

## Pointing error

Optional model of the error in directing a satellite at its target, following
`Target_Noise_Model_Document.pdf` (ECSS-E-ST-60-10C / ESA ESSB-HB-E-003). Per
satellite, per camera axis (x, y across the boresight; z about it):

`e = bias + per-acquisition offset + Gauss-Markov drift + jitter`

| Term | Model | Config keys |
|---|---|---|
| Static bias | drawn once per satellite per run, σ_b | `sigma_bias_deg` |
| Per-acquisition offset | redrawn every render, σ_p | `sigma_acquisition_deg` |
| Drift | first-order Gauss-Markov, σ_g, time constant τ | `sigma_drift_deg`, `drift_tau_s` |
| Jitter | white Gaussian per render, σ_j | `sigma_jitter_deg` |

All sigmas are 1σ per cross-boresight axis; `yaw_factor` multiplies the about-boresight
(z) axis (document: 2–5×). `acquisition_interval_s` is the time between successive
renders, which sets how far the drift evolves from one render to the next.

The error rotates the commanded line of sight. The rotated line is ray-traced to the
ground, the image footprint moves to where it actually lands (and turns by the yaw
error), and the image is rendered along the actual line of sight. The cloud field stays
centred on the commanded target. With a periodic domain the shifted pixels wrap around;
with a non-periodic one they may enter the cloud-free margin, and a shift beyond the
domain (about ±33 km, see the oblique-views note below) raises an error.

Presets (the document's illustrative starting values; engineering assumptions, not
published figures): *6U CubeSat (C3IEL-like, ~0.1°)* and *Agile EO satellite (~0.01°)*.
Editing any sigma in the GUI switches the preset to *Custom*.

### Random state and reproducibility

- `noise_seed` seeds the run. In the GUI, **New Monte-Carlo run** advances the seed
  (new static biases and drift for every satellite).
- Each render is one *acquisition*: bias persists for the whole run; the per-acquisition
  offset and jitter are redrawn; the drift carries over with correlation
  `exp(-Δt/τ)`.
- A headless run resets the noise to `noise_seed` and does acquisitions 1…`renders`.
  The GUI after a seed reset does the same on successive **Render** clicks, so the
  headless run draws the same pointing errors the GUI showed.
- Images are **not** byte-identical between runs: SMART-G's own photon noise is not
  seeded. Only the pointing errors are reproducible.
- If the pointing error is off, `renders > 1` is ignored (one render), because repeating
  a noise-free render only rewrites the same file.

### Where the satellites look (oblique views)

All satellites aim at one point: the ground target raised to the **cloud field's centre
height** (1.72 km, halfway between the lowest cloud base and the highest cloud top of the
field), not at the ground. SMART-G's sensors sit at the top of the 3D domain (30 km), so an
oblique line of sight drifts sideways by `(30 - 1.72) km × tan(VZA)` on its way down to the
clouds (24.6 km at VZA 41°, the formation's outermost satellite). Each satellite's pixel
plane is shifted back by that amount, so every satellite's central ray crosses the cloud
field's centre at cloud height and they all see the same cloud. Aiming at the ground instead
left most satellites of the formation looking at empty sky beside the field. The cloud-free
margin around the field is 30 km wide (`main.SMARTG_3D_HORIZ_EXTEND_KM`) so that these
shifted footprints fit.

### Output names

Single: `single_offnadir<N>_az<N>_sza<N>_saa<N>_<periodic|nonperiodic>[_ocean<W>ms][_noise_seed<S>_render<k>].png` (+ `.npy`)

Formation: a folder `formation_sza<N>_saa<N>_<periodic|nonperiodic>[_ocean<W>ms][_noise_seed<S>_render<k>]/`
containing `satellite_<NN>.png` (+ `.npy`, `_raw.png`) per rendered satellite.

Each image also gets `<name>_raw.png`: one heat map per wavelength of the raw reflectance
on a shared 0-to-max colour scale with colour bars (no gain, gamma or clipping), and the log
prints, per satellite and wavelength, the raw min / median / 99th percentile / max and the
share of pixels the display gain clips.

Each PNG has a `.npy` file of the same name next to it holding the raw reflectance, before
the display gain, gamma and clipping that turn it into the PNG (which saturates bright
clouds). It is a float array of shape `(height, width, 3)`, rows top to bottom, with the
last axis in the order of `main.SMARTG_WAVELENGTHS_NM` - 440, 550, 670 nm, i.e. **B, G, R**,
not RGB. Load it with `numpy.load`; to rebuild an image with another gain:

```python
import numpy as np, main
r = np.load("smartg_playground_images/<run>/satellite_04.npy")
main.reflectance_to_image(r, gain=2.0, gamma=1.8).save("satellite_04_gain2.png")
```

The `_noise_seed…_render…` suffix is added when the pointing error is on, so renders of
a run don't overwrite each other. The status text / printed log gives each satellite's
line-of-sight error, ground offset (east/north, km) and yaw.

## Setting up the server

Needs a Linux machine with an NVIDIA GPU. This project can't run SMART-G on CPU.

1. Clone the repository.
2. Python 3.11–3.14 with the project's libraries (numpy, xarray, scipy, pandas, h5py,
   skyfield, requests, Pillow, traits/traitsui/pyface, mayavi/vtk, …).
3. **CUDA**: NVIDIA driver, CUDA toolkit (`nvcc` on `PATH`), a host C/C++ compiler
   supported by that `nvcc` version, and **PyCUDA** (`pip install pycuda`). Check:
   `python -c "import pycuda.driver as d; d.init(); print(d.Device(0).name())"`.
4. **SMART-G** from HYGEOS' GitHub:
   `pip install "git+https://github.com/hygeos/smartg.git@release/2.0.0"`
   (version 2.0.0b1; pulls in geoclide, luts, pytrunc, gatiab, …). SMART-G requires
   numpy ≥ 2, < 3.
5. **SMART-G auxiliary data** (~535 MB, git-ignored), from the repository root:
   `python -c "import main; main.ensure_smartg_auxdata(data_type='all')"`
   Check that `smartg_auxdata/IPRT/phaseB/grids/cumulus.dat` exists.
6. Set `CONFIG_PATH` and run `python run_from_config.py`. The first run compiles the
   CUDA kernels and is slow.

Mayavi and Traits must be *importable* on the server (the headless run still imports
`main.py`) but no display is needed; see the next section.

### How headless imports work

`run_from_config.py` sets `ETS_TOOLKIT=null` before importing, so Traits/Pyface never try
to start Qt. `main.py` builds its (here unused) Mayavi window description at import time,
and the no-GUI backend can't build the 3D scene editor for it; on that failure the runner
swaps in a placeholder editor and imports again. Nothing in the headless path opens a
window. If a real Qt backend is available, the first import succeeds and the placeholder
is not used.

`main.py` contains some Windows-specific code for the CUDA host compiler
(`ensure_cuda_compatible_msvc_on_path`, `_find_vs2022_msvc_bin_dir`). It is documented as
a no-op when no Visual Studio install is found, so it should do nothing on Linux. If it
raises there, make it return early when `sys.platform != "win32"`.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `Could not load …: …` (exit 2) | Config missing, wrong `CONFIG_PATH`, or it failed validation; the message names the key. |
| `ModuleNotFoundError: smartg` | SMART-G not installed in this Python environment. |
| `nvcc fatal: Unsupported gcc version` | Install a host compiler supported by the CUDA toolkit and point `nvcc` at it (`-ccbin`). |
| PyCUDA can't find `cuda.h` / `libcuda` | Set `CUDA_HOME`, add `$CUDA_HOME/bin` to `PATH` and `$CUDA_HOME/lib64` to `LD_LIBRARY_PATH`. |
| `FileNotFoundError … cumulus.dat` | Auxiliary data not downloaded; run step 5 of the server setup. |
| `the image footprint was moved past the simulation domain's edge` | The pointing-error sigmas and/or the oblique view are too large for a non-periodic domain (limit ≈ ±33 km; the error text gives both contributions); reduce them or use a periodic domain. |
| `NotImplementedError … doesn't implement scene_editor` | Only if the import workaround in `run_from_config.py` was removed. |
| Loaded config's geometry "was not applied" (GUI) | Expected for the in-window **Load** button; start the GUI with the config path to restore it. |

## Limitations

- Knowledge error (what the satellite believes it points at) is not modelled; it only
  matters for geolocating images afterwards.
- Jitter is one random sample per render. Sinusoidal reaction-wheel jitter and
  intra-exposure blur are not modelled: each render is one frozen exposure.
- The in-window **Load configuration…** applies settings, not geometry.
- Only the 3D sandbox renderer is covered. `main.py`'s own animated simulation has no
  headless mode.
