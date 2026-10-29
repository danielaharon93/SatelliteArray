# Cloud-free ocean: comparing the simulation with MODIS Terra

The aim: simulate a cloud-free ocean scene (molecular atmosphere, optional aerosol, Cox-Munk
ocean) for the sun/view geometry of a real MODIS pixel, and compare the top-of-atmosphere
reflectance in MODIS bands b03 (blue), b04 (green), b01 (red). Run it with `run_clear_ocean.py`
(settings in `configs/clear_ocean_template.json`).

## Candidate scenes (South Pacific gyre, box 125W-115W, 28S-22S)

Chosen from NASA GIBS daily true-colour mosaics of every day of 2023 (fewest bright / glint
pixels in the box), then looked at by eye. These are **candidates**: the mosaics still show
scattered small cumulus, so the final choice needs the cloud mask (MOD35) of the granule.

| Date | Terra granules over the box (UTC) | Notes |
|---|---|---|
| 2023-03-05 (day 064) | 17:50, 17:55 | Darkest box; a glint patch near its east edge |
| 2023-04-03 (day 093) | 18:00, 18:05 | Clear patch in the centre; some cloud lines at the south |

Files (download needs a free Earthdata Login; replace `<product>` / names below):
`https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/<product>/<year>/<day-of-year>/<file>`

| Product | What for | 2023-03-05 17:50 | 2023-04-03 18:00 |
|---|---|---|---|
| MOD021KM | radiances (observed reflectance) | MOD021KM.A2023064.1750.061.2023065012217.hdf | MOD021KM.A2023093.1800.061.2023094074422.hdf |
| MOD03 | per-pixel sun/view angles, lat/lon | MOD03.A2023064.1750.061.2023064231813.hdf | MOD03.A2023093.1800.061.2023094012102.hdf |
| MOD04_L2 | aerosol optical depth | MOD04_L2.A2023064.1750.061.2023132222433.hdf | MOD04_L2.A2023093.1800.061.2023134004933.hdf |
| MOD35_L2 | cloud mask | MOD35_L2.A2023064.1750.061.2023132005504.hdf | MOD35_L2.A2023093.1800.061.2023133091328.hdf |
| MOD07_L2 | ozone, water vapour, pressure | MOD07_L2.A2023064.1750.061.2023132005525.hdf | MOD07_L2.A2023093.1800.061.2023133091349.hdf |

(The 17:55 / 18:05 granules follow them; their names are in NASA's catalogue.)

## What to put in the scenario file

| Field | Take it from |
|---|---|
| `sun_zenith_deg`, `sun_azimuth_deg`, `view_zenith_deg`, `view_azimuth_deg` | MOD03 `SolarZenith`, `SolarAzimuth`, `SensorZenith`, `SensorAzimuth` of the chosen pixel (divide the stored values by 100) |
| `latitude_deg` | MOD03 `Latitude` |
| `aod_550` | MOD04_L2 ocean AOD at 550 nm over the pixel's 10 km cell |
| `ozone_dobson`, `water_vapour_g_cm2`, `surface_pressure_hpa` | MOD07_L2 (or MERRA-2 / ERA5) |
| `wind_speed_ms` | MERRA-2 / ERA5 10 m wind; matters little away from the glint |
| `observed_reflectance` (`b03`, `b04`, `b01`) | MOD021KM reflectance of the pixel (average cloud-free pixels around it); check the user guide whether the stored reflectance already includes the division by cos(sun zenith) |

The template's geometry is only approximate (sun from a formula, view angles placeholders).

## What the simulation includes, and what it leaves out

* Included: Rayleigh scattering, gas absorption (AFGL profile, scaled to the ozone / water vapour /
  pressure you give), an OPAC aerosol of the given optical depth, a Cox-Munk rough sea (reflection
  and glint, wave shadowing).
* Not included: light that comes out of the water (ocean colour, strongest in blue, about zero in
  red) and whitecaps. Compare the red channel (b01) first; expect the simulated blue (b03) to be low.
* The atmosphere profile defaults to `afglt` (tropical) in the template; `afglus` is what the cloud
  runs use.

## First result (scene 2023-03-05 17:50, box 125W-115W, 28S-22S)

Observed: median TOA reflectance of 95,463 cloud-free, glint-free ocean pixels. Simulated:
`run_clear_ocean.py`, 1e8 photons, 15 x 30 angular bins (repeated runs agree to about 1%). Aerosol
0.087 (MOD04), ozone 263 DU, water vapour 3.05 g/cm2 and pressure 1019 hPa (MOD07), wind 4 m/s
(placeholder), `maritime_clean` aerosol (a guess).

| | b03 blue | b04 green | b01 red |
|---|---|---|---|
| observed | 0.1239 | 0.0541 | 0.0292 |
| placeholder atmosphere (AOD 0.06), reflecting sea | 0.77 | 0.86 | 0.89 |
| real AOD / ozone / water vapour / pressure, reflecting sea | 0.79 | 0.88 | 1.00 |
| ... plus water body, chlorophyll 0.02 | 0.92 | 0.93 | 0.98 |
| ... plus water body, chlorophyll 0.05 | 0.89 | 0.95 | 0.98 |
| ... plus water body, chlorophyll 0.10 | 0.87 | 0.94 | 0.98 |

(simulated / observed.) The water body costs about 50 times the run time (250 s against 5 s at
1e8 photons). Left unexplained: blue 8-13% and green 5-7% low; the wind, aerosol type, whitecaps and the
cloud screening (a near-infrared test, not MOD35) were not varied. That MODIS "reflectance" has to be divided by cos(sun zenith) is
from memory, not checked in the user guide.

## In the playground (3D cloud render)

* Aerosol, ozone, water vapour and pressure are playground settings now (`aerosol_model`, `aod_550`, `ozone_dobson`,
  `water_vapour_g_cm2`, `surface_pressure_hpa`; the defaults are the old behaviour: no aerosol, the profile's own gases).
* With an aerosol SMART-G needs a phase function per optical cell (about 20,000 for the bundled cloud) and tries to
  store each at 1e6 angles (1.3 TB), so the phase table is set to 721 angles (`AEROSOL_N_ICDF`). In a small test this
  gave no visible change of the mean reflectance beyond the noise, and was faster; it was not tested more rigorously.
* The grid band set with an aerosol did not finish: after more than 20 minutes it was still building phase tables
  (30 wavelengths x about 20,000 cells) and had used 9.6 GB of host memory, so it was stopped. The BOMEX field has more cells.
* The water body is not available in the 3D render (SMART-G fails with `KeyError: 'iabs_oc'`); only `run_clear_ocean.py` has it.

Note: the water-body rows above were run with the `afglus` profile (the scenario's `atmosphere_profile` was
not passed on at that point; fixed since, it is `afglt` now). A reflecting-sea run with `afglt` after the fix gave
0.79 / 0.91 / 0.98, the same as with `afglus` within the 1-4% noise, but the water-body runs were not repeated with `afglt`.
