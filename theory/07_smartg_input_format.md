# 7. What SMART-G accepts: the input data format

This is an implementation reference, not theory. It records how the 3D scene
that SMART-G renders is organised, as inferred from the playground's original
pipeline (the bundled `cumulus.dat` cloud) and SMART-G's own docstrings, so
that a cloud built from the BOMEX dataset ([cloud_microphysics.py](../cloud_microphysics.py))
can be handed to SMART-G in a form it handles.

How it was found: the files SMART-G opens during a render were traced with a
Python audit hook (Python-level opens only), the data files were opened and
read, and the `Cloud3D`, `Grid3D`, `Atm3D` and `read_i3rc_cloud` docstrings
and source were read (SMART-G 2.0.0b1, Anaconda base environment). Findings
that come from a test render rather than from the documentation say so.

## 7.1 The scene and the files behind it

A 3D render (`Smartg(back=True, opt3d=True)`) is built from four objects:

```
Atm3D ──┬── Atm1D("afglus", lat)      background atmosphere: molecules, gases
        ├── Grid3D(x, y, z bounds)    the 3D cells
        └── [Cloud3D(...)]            the 3D cloud (cells that hold droplets)
surface   LambSurface or RoughSurface (no data files)
sensor + LocalEstimate                where the camera is, and the Sun direction
```

Auxiliary files opened (identical for the Lambertian and the ocean surface):

| File | Role |
|---|---|
| `IPRT/phaseB/grids/cumulus.dat` | the 3D cloud: per-cell extinction and effective radius |
| `clouds/wc_sol.nc` | droplet optics indexed by effective radius: extinction vs wavelength, single-scattering albedo, phase matrices |
| `atmospheres/afglus.nc` | AFGL US-standard atmosphere profile |
| `acs/O3_acs_BogumilV3.0_coeffs.nc` | ozone absorption cross-section coefficients |
| `acs/NO2_acs_BogumilV1.0_coeffs.nc` | NO2 absorption cross-section coefficients |

Not opened: the ice-cloud tables (`clouds/ic_baum_*`), the other atmosphere
profiles, the other `IPRT` folders, any aerosol file. The ocean surface needs no
file. Rayleigh scattering is computed from the profile, not read.

The directory comes from the environment variable `SMARTG_DIR_AUXDATA`
(`main.SMARTG_AUXDATA_DIR`, next to `main.py`, in the playground).

## 7.2 The 3D cloud: what `Cloud3D` takes

`Cloud3D(fname, w_ref, ds, reff_acc, reff_min, ...)` takes the cloud as an
`xarray.Dataset` ("dense schema"):

| Item | Shape | Unit | Meaning |
|---|---|---|---|
| `ext` | `(z, y, x)` | km⁻¹ | extinction coefficient at `w_ref`; **0 in cloud-free cells** |
| `reff` | `(z, y, x)` | µm | droplet effective radius |
| `x_bounds` | `(x_b)` = nx + 1 | km | cell boundaries, ascending |
| `y_bounds` | `(y_b)` = ny + 1 | km | cell boundaries, ascending |
| `z_bounds` | `(z_b)` = nz + 1 | km | cell boundaries, ascending; not required to be uniform |
| attribute `w_ref` | scalar | nm | wavelength of `ext`; or pass `w_ref=` to the constructor |

Dimension names must be exactly `z`, `y`, `x`, and the three `*_bounds` must be
coordinates. A missing name raises a `ValueError` that lists it.

What `Cloud3D` does with the dataset (from its source):
- **Only cells with `ext > 0` are cloud.** They are extracted in C order with x
  slowest, and `reff` is read at those cells only. The `reff` value in clear
  cells is ignored (it is 0 in `cumulus.dat`).
- `reff_acc` rounds `reff` to that many decimals (the playground uses 1);
  `reff_min` / `reff_max` clamp it.
- `reff` is looked up in the table `fname` (`"wc"` = `wc_sol.nc`) by linear
  interpolation. Out-of-range values **raise an error** (the table covers
  5-30 µm), so `reff_min` must be at least the table's minimum (5 µm).
- `ext` is at `w_ref`. For each wavelength SMART-G multiplies it by the table's
  `ext(reff, λ) / ext(reff, w_ref)`, about 1% between 440 and 670 nm.
- The phase matrices and the single-scattering albedo come from the table,
  unless overridden: `phase=` takes a `DataArray` with dims
  `(wavelength_phase, reff, nphamat, theta_atm)` (4 terms are completed to 6),
  `ssa_cst=` forces the albedo.

### The bundled cloud, as a reference case

`read_i3rc_cloud(cumulus.dat)` returns that dataset. Its ASCII format is: one
comment line; `Nx Ny Nz flag`; `Dx Dy z_0 z_1 … z_Nz` (z bounds in km); then one
line per **cloudy cell only** with 1-based `ix iy iz`, extinction (km⁻¹) and
`reff` (µm). x and y bounds are built centred on 0 (`loc="centered"`). Measured
on the file (the BOMEX field, from `cloud_microphysics.py`, for comparison):

| | `cumulus.dat` | BOMEX `int_2` file |
|---|---|---|
| Grid (z, y, x) | 53 × 100 × 100 = 530,000 | 170 × 512 × 512 = 44.6 million |
| Horizontal spacing | 0.0667 km | 0.02 km |
| Horizontal extent | 6.67 km × 6.67 km | 10.24 km × 10.24 km |
| z bounds | 0 to **30 km**, layers 0.2 km below 1 km, 0.04 km from 1.0 to 2.44 km, then 0.16, 0.2, 0.5, 1, 2, 3, 5 km up to 30 km | 0 to 3.985 km, 0.02 km up to 3 km then widening to 0.05 km |
| Cloudy cells | 20,435 (3.9%) | 363,946 (0.8%) |
| `ext` (cloudy) | 0.04 / 68 / 196 km⁻¹ (min / median / max) | 3.4 / 33 / 273 km⁻¹ (p5 / median / max) |
| `reff` (cloudy) | 4.0 / 8.5 / 14.4 µm | 4.3 / 7.6 / 17.5 µm (p5 / median / max) |
| Max vertical optical depth | 151 | 144 (Q = 2) |

## 7.3 The grid: `Grid3D`

`Grid3D(xgrid, ygrid, zgrid, periodic, horiz_extend_length, vert_extend_limit)`
takes three 1D **numpy arrays**, sorted ascending, of the cell boundaries in
km. The playground passes the cloud dataset's `x_bounds`, `y_bounds`,
`z_bounds`.

- **Horizontal boundary:** either `periodic=True` (photons leaving one side
  re-enter the other), or a hard edge with `horiz_extend_length` km of empty
  margin on every side (the playground uses 30 km). The two are mutually
  exclusive.
- **Sensors:** the playground puts them at `grid_3d.zGRID[-1]`, the top of the
  *whole* domain (with a 1e-6 km offset), and spreads the pixels over
  `xgrid`/`ygrid`. This is the "top of the atmosphere" for `LocalEstimate`
  with `count_level=0`.
- **Atmosphere above the cloud:** the 1D background is evaluated on the grid's
  own vertical layers, so it exists only up to the grid top. `cumulus.dat`
  therefore carries empty, coarse layers up to 30 km, which hold the molecules
  and the ozone above the cloud.

### Finding: the grid top has to reach about 30 km

I tested this with a cropped BOMEX cloud (200 × 200 × 120 cells, 2.4 km tall),
Henyey-Greenstein phase, 40 × 40 pixels, 20,000 photons per pixel, nadir view,
Sun at 30°, Lambertian surface (albedo 0.05). The mean reflectance in
440 / 550 / 670 nm, and the clear-sky level (10th percentile of the pixels):

| Grid top | Mean reflectance | Clear-sky level |
|---|---|---|
| 2.4 km (the cloud grid, unpadded) | 0.096 / 0.082 / 0.078 | 0.034 / 0.021 / 0.016 |
| 30 km, coarse empty layers at 3, 4, 5, 7, 10, 15, 20, 25, 30 km | 0.143 / 0.098 / 0.085 | 0.099 / 0.046 / 0.027 |
| 120 km, one extra cell via `vert_extend_limit=120` | 0.032 / 0.013 / 0.006 | 0.029 / 0.012 / 0.005 |

- Without the layers above the cloud, the Rayleigh scattering and the ozone
  above 2.4 km are missing. The clear-sky blue-to-red ratio is then 2.2,
  against 3.7 with the padding; Rayleigh scattering varies as λ⁻⁴, which
  favours the padded result.
- Padding with several empty layers up to 30 km behaves sensibly, as
  `cumulus.dat` does.
- A single 117 km cell from `vert_extend_limit` gave much darker images (670 nm
  down by a factor of about 13). I did not find out why, so **do not use
  `vert_extend_limit`**; pad explicitly instead.
- This is one small, low-photon test, so the numbers show the size of the
  effect, not a calibrated result.

## 7.4 The droplet table: `clouds/wc_sol.nc`

| Variable | Dimensions | Content |
|---|---|---|
| `phase` | `(reff, wav, stk, theta)` | phase matrix; `stk` = P11, P21, P33, P34; P11 normalised so that ∫P11 sinθ dθ = 2 |
| `ext` | `(reff, wav)` | labelled "extinction coefficient in km⁻¹" |
| `ssa` | `(reff, wav)` | single-scattering albedo |
| coordinates | `reff` 26 values, 5-30 µm in 1 µm steps; `wav` 209 values, 253-4441 nm; `theta` 594 values, 0-180°, non-uniform | |
| attribute | `veff = 0.1` | effective variance of the underlying size distribution |

- With P22 = P11 and P44 = P33 completed by SMART-G, the 4 terms describe
  spherical particles.
- At 670 nm and `reff` = 10 µm the asymmetry parameter is 0.861 and the albedo
  0.999996.
- The `ext` values behave like a per-droplet extinction cross-section in µm²,
  not like a coefficient per g/m³: they scale as `reff²` (128, 496, 1950, 4354
  for 5, 10, 20, 30 µm), and are a constant 1.044 times the cross-section of a
  gamma distribution with α = 7 that I integrated with Mie theory. That is my
  inference, and the 1.044 is unexplained. SMART-G only uses the ratio between
  wavelengths, so the unit does not enter the render.
- The table is probably libRadtran `mie` output (gamma distribution α = 7
  gives effective variance 1/(α+3) = 0.1), but nothing in the file says so.

## 7.5 The background atmosphere

`Atm1D("afglus", lat=…)`:
- `atmospheres/afglus.nc`: dimension `z_atm` (50 levels, 120 km down to 0 km,
  descending); variables `P`, `T`, `dens`, `H2O`, `O3`, `N2O`, `CH4`, `CO2`,
  `O2`, `N2`, `NO2`, `SO2`; attribute `name` "U.S. Standard (1976)".
- `acs/*_coeffs.nc`: dimension `wav` (O3: 8364 values, 230-1066 nm); variables
  `C0`, `C1`, `C2`; cross-section σ = 10⁻²⁰ · (C0 + C1·T + C2·T²) cm², with T
  from the profile (the file's `T` coordinate lists 203-293 K).
- `lat` (degrees) only enters the Rayleigh calculation.

`Atm3D` evaluates this profile on the 3D grid's vertical layers. In the cells
that hold cloud it adds the cloud's extinction to the background's, weights the
single-scattering albedo by extinction, and the phase matrices by the
scattering coefficients.

## 7.6 Checklist for a BOMEX cloud

| Requirement | State |
|---|---|
| `ext(z, y, x)` in km⁻¹ at 670 nm, 0 in clear cells | `CloudOptics.beta` (Eq. 3; `q2` or `mie`); done by `to_smartg_dataset` |
| `reff(z, y, x)` in µm | `CloudOptics.r_eff`; done |
| dimension names `z`, `y`, `x`, bounds as coordinates in km | done |
| `w_ref` attribute | done (670 nm) |
| `reff` within 5-30 µm | `reff_min=5` clamps the 11% of cloudy voxels below it; the BOMEX maximum (17.5 µm) is within range |
| arrays in `(z, y, x)` | native order, no transpose |
| **grid top near 30 km with coarse empty layers** | **not yet done**: `to_smartg_dataset` stops at the data's own top (3.985 km, or lower if cropped) |
| horizontal boundary | `Grid3D(…, periodic=…, horiz_extend_length=30)` as in the playground |
| sensor height | `grid.zGRID[-1] - 1e-6` of the padded grid |
| size | 44.6 million cells for the full file against 0.53 million for `cumulus.dat`; only a crop has been rendered (200 × 200 × 120) |

So the one missing step is padding the vertical grid: append layers with
`ext = 0` at about 4, 5, 7, 10, 15, 20, 25, 30 km (any coarse set works) in
`to_smartg_dataset`, then build `Grid3D` from the padded bounds. The padded
dataset I used in the test above was built that way.

## 7.7 Not yet established

- Whether SMART-G's compiled code reads files that the Python-level trace
  could not see.
- Why the single extra cell from `vert_extend_limit` gives dark images.
- The origin of the 1.044 factor between the table's `ext` and an integrated
  Mie cross-section.
- Memory and run time for the full 512 × 512 domain.
