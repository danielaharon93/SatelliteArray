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
| `resolution` | `"Tiny (16x16)"`, `"Small (32x32)"` or `"Medium (64x64)"`. |
| `photons_per_pixel` | Photons per pixel; total sent to SMART-G = this × pixel count. |
| `pointing_noise_enabled` | Turn the pointing-error model on. |
| `sigma_bias_deg`, `sigma_acquisition_deg`, `sigma_drift_deg`, `drift_tau_s`, `sigma_jitter_deg`, `yaw_factor`, `acquisition_interval_s` | Pointing-error parameters (below). |
| `noise_seed` | Random seed of the Monte-Carlo run. |
| `renders` | Headless only: number of acquisitions to render. |

The file can be edited by hand. On load it is validated: unknown keys, an unknown
mode/resolution, out-of-range angles, bad satellite indices, `renders < 1`, a
wrong `version` and an unparseable `time_utc` are all rejected with a message.

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
domain (about ±23 km) raises an error.

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

### Output names

Single: `single_offnadir<N>_az<N>_sza<N>_saa<N>_<periodic|nonperiodic>[_noise_seed<S>_render<k>].png`

Formation: a folder `formation_sza<N>_saa<N>_<periodic|nonperiodic>[_noise_seed<S>_render<k>]/`
containing `satellite_<NN>.png` per rendered satellite.

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
| `pointing error moved the image footprint … past the simulation domain's edge` | The sigmas are too large for a non-periodic domain (limit ≈ ±23 km); reduce them or use a periodic domain. |
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
