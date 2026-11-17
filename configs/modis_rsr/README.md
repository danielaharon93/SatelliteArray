# MODIS relative spectral responses (bands 3, 4, 1)

`modis_terra_rsr_b03_b04_b01.csv` and `modis_aqua_rsr_b03_b04_b01.csv`: columns `nm, blue (b03), green (b04), red (b01)`,
0.5 nm spacing, each band peak-normalised to 1.

Source: NASA MODIS Characterization Support Team, https://mcst.gsfc.nasa.gov/calibration/parameters
(`Terra_RSR_in-band.xlsx`, `Aqua_RSR_in-band.xlsx`: in-band RSR of Terra PFM and Aqua FM1, given where the response is at least 1 % of the peak).
Made by averaging the detector channels of each band (the workbooks' -99 values are missing data and were dropped).
Only the in-band part is included: the response below 1 % of the peak is not.

Used by `process_reflectance_gui.py` (Channels: "MODIS Terra / Aqua response"). The channels are rebuilt from the
per-wavelength data, so only the wavelengths that were simulated count: with the `grid_modis` band set that is the
REPTRAN limits of each band, a little narrower than the 1 % edges of these curves.
