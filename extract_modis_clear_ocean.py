"""Read a MODIS Terra L1B granule (MOD021KM) and make a scenario file for run_clear_ocean.py.

It picks the cloud-free, glint-free ocean pixels of the granule inside a lat/lon
box, takes their median top-of-atmosphere reflectance in MODIS bands b03, b04 and
b01 and their mean sun and view geometry, and writes them into a copy of
configs/clear_ocean_template.json. The aerosol, ozone, water vapour, pressure and
wind of the template are NOT in this file: take them from MOD04_L2 / MOD07_L2 /
MERRA-2 and edit the scenario file yourself.

    python extract_modis_clear_ocean.py

Needs pyhdf (reads HDF4) and scipy.
"""

import json
import sys
from pathlib import Path

import numpy as np

# ============================== SETTINGS ==============================
# A relative path is taken relative to this file's folder.
MOD021KM_FILE = "MOD021KM.A2023064.1750.061.2023065012217.hdf"
TEMPLATE = "configs/clear_ocean_template.json"
OUTPUT_DIR = "configs"

# The aerosol (MOD04_L2) and atmosphere (MOD07_L2) files of the same granule; set to
# None to leave the template's values. Their median inside BOX is used.
MOD04_FILE = "MOD04_L2.A2023064.1750.061.2023132222433.hdf"
MOD07_FILE = "MOD07_L2.A2023064.1750.061.2023132005525.hdf"

# Pixels are used if they lie in this box (degrees; west, south, east, north) ...
BOX = (-125.0, -28.0, -115.0, -22.0)
# ... look at the ground within this angle of the vertical (the viewing geometry is
# then close to one value) ...
MAX_VIEW_ZENITH_DEG = 25.0
# ... are further than this from the specular (sun glint) direction ...
MIN_GLINT_ANGLE_DEG = 40.0
# ... and are dark in the near infrared, where the ocean reflects almost nothing and
# a cloud, glint or whitecap does: band 2 (859 nm) and band 26 (1.38 um, cirrus).
# These are reflectance values divided by cos(sun zenith). The whole 3 x 3 block
# around a pixel has to pass, so cloud edges are left out.
MAX_B02_REFLECTANCE = 0.03
MAX_B26_REFLECTANCE = 0.01
# =======================================================================

here = Path(__file__).parent


def read_sds(sd, name: str) -> np.ndarray:
    return sd.select(name).get()


def scaled(sd, name: str) -> np.ndarray:
    """A geolocation / angle dataset as float degrees, fill values as NaN."""
    sds = sd.select(name)
    raw = sds.get().astype(float)
    attrs = sds.attributes()
    raw[raw == attrs["_FillValue"]] = np.nan
    return raw * attrs.get("scale_factor", 1.0) + attrs.get("add_offset", 0.0)


def to_pixels(tie: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Interpolate a 5 x 5-subsampled geolocation field to the 1 km pixel grid."""
    from scipy.ndimage import zoom

    return zoom(np.nan_to_num(tie, nan=np.nanmean(tie)), (shape[0] / tie.shape[0], shape[1] / tie.shape[1]), order=1)


def reflectance(sd, dataset: str, index: int | None = None) -> np.ndarray:
    """Reflectance times cos(sun zenith) of a band, as MODIS stores it (reflectance_scales *
    counts); invalid counts are NaN."""
    sds = sd.select(dataset)
    counts = sds.get()
    attrs = sds.attributes()
    scale = np.atleast_1d(attrs["reflectance_scales"])
    offset = np.atleast_1d(attrs["reflectance_offsets"])
    i = 0 if index is None else index
    plane = (counts if index is None else counts[index]).astype(float)
    plane[(counts if index is None else counts[index]) > 32767] = np.nan
    return scale[i] * (plane - offset[i])


def box_median(sd, dataset: str) -> tuple[float, int, tuple[float, float]]:
    """Median of a geolocated (lat/lon on the same grid) dataset over the cells inside BOX,
    with the number of valid cells and their 10th / 90th percentiles."""
    west, south, east, north = BOX
    lat, lon = scaled(sd, "Latitude"), scaled(sd, "Longitude")
    sds = sd.select(dataset)
    attrs = sds.attributes()
    values = sds.get().astype(float)
    values[values == attrs["_FillValue"]] = np.nan
    values = values * attrs.get("scale_factor", 1.0) + attrs.get("add_offset", 0.0)
    inside = values[(lat >= south) & (lat <= north) & (lon >= west) & (lon <= east)]
    inside = inside[np.isfinite(inside)]
    if inside.size == 0:
        sys.exit(f"No valid {dataset} values inside BOX.")
    return float(np.median(inside)), int(inside.size), (float(np.percentile(inside, 10)), float(np.percentile(inside, 90)))


def read_ancillary() -> dict:
    """AOD at 550 nm (MOD04_L2, over ocean), total ozone (DU), water vapour (cm, equal to
    g/cm2) and surface pressure (hPa) (MOD07_L2): medians over BOX."""
    from pyhdf.SD import SD, SDC

    out, notes = {}, {}
    if MOD04_FILE:
        sd = SD(str(here / MOD04_FILE), SDC.READ)
        out["aod_550"], n, spread = box_median(sd, "Effective_Optical_Depth_0p55um_Ocean")
        notes["aod_550"] = {"cells": n, "p10_p90": spread}
    if MOD07_FILE:
        sd = SD(str(here / MOD07_FILE), SDC.READ)
        for key, dataset in (("ozone_dobson", "Total_Ozone"), ("water_vapour_g_cm2", "Water_Vapor"), ("surface_pressure_hpa", "Surface_Pressure")):
            out[key], n, spread = box_median(sd, dataset)
            notes[key] = {"cells": n, "p10_p90": spread}
    for key in out:
        out[key] = round(out[key], 4)
    return out, notes


def circular_mean_deg(angles_deg: np.ndarray) -> float:
    rad = np.radians(angles_deg)
    return float(np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360.0)


def main() -> int:
    from pyhdf.SD import SD, SDC
    from scipy.ndimage import minimum_filter

    path = here / MOD021KM_FILE  # an absolute path wins
    if not path.is_file():
        sys.exit(f"{path} does not exist: set MOD021KM_FILE at the top of {Path(__file__).name}")
    sd = SD(str(path), SDC.READ)

    b01 = reflectance(sd, "EV_250_Aggr1km_RefSB", 0)
    b02 = reflectance(sd, "EV_250_Aggr1km_RefSB", 1)
    b03 = reflectance(sd, "EV_500_Aggr1km_RefSB", 0)
    b04 = reflectance(sd, "EV_500_Aggr1km_RefSB", 1)
    b26 = reflectance(sd, "EV_Band26")
    shape = b01.shape

    lat, lon = (to_pixels(scaled(sd, n), shape) for n in ("Latitude", "Longitude"))
    sza, saa, vza, vaa = (to_pixels(scaled(sd, n), shape) for n in ("SolarZenith", "SolarAzimuth", "SensorZenith", "SensorAzimuth"))

    cos_sza = np.cos(np.radians(sza))
    b02r, b26r = b02 / cos_sza, b26 / cos_sza
    dark = np.nan_to_num(b02r, nan=1.0) < MAX_B02_REFLECTANCE
    dark &= np.nan_to_num(b26r, nan=1.0) < MAX_B26_REFLECTANCE
    dark = minimum_filter(dark.astype(np.uint8), size=3).astype(bool)  # the 3 x 3 block must all be dark

    cos_glint = np.cos(np.radians(sza)) * np.cos(np.radians(vza)) - np.sin(np.radians(sza)) * np.sin(np.radians(vza)) * np.cos(np.radians(saa - vaa))
    glint = np.degrees(np.arccos(np.clip(cos_glint, -1.0, 1.0)))

    west, south, east, north = BOX
    in_box = (lat >= south) & (lat <= north) & (lon >= west) & (lon <= east)
    valid = np.isfinite(b01 + b03 + b04)
    chosen = in_box & valid & (vza <= MAX_VIEW_ZENITH_DEG) & (glint >= MIN_GLINT_ANGLE_DEG) & dark

    print(f"{path.name}\n  pixels in the box: {in_box.sum()}; "
          f"within view zenith {MAX_VIEW_ZENITH_DEG:g}: {(in_box & (vza <= MAX_VIEW_ZENITH_DEG)).sum()}; "
          f"also glint-free: {(in_box & (vza <= MAX_VIEW_ZENITH_DEG) & (glint >= MIN_GLINT_ANGLE_DEG)).sum()}; "
          f"also dark in the near infrared (used): {chosen.sum()}")
    if chosen.sum() < 100:
        sys.exit("Too few usable pixels: widen BOX / MAX_VIEW_ZENITH_DEG or relax the near-infrared limits.")

    observed, spread = {}, {}
    for name, band in (("b03", b03), ("b04", b04), ("b01", b01)):
        values = (band / cos_sza)[chosen]
        observed[name] = round(float(np.median(values)), 5)
        spread[name] = {"std": round(float(values.std()), 5), "p10": round(float(np.percentile(values, 10)), 5), "p90": round(float(np.percentile(values, 90)), 5)}

    geometry = {
        "latitude_deg": round(float(lat[chosen].mean()), 3),
        "sun_zenith_deg": round(float(sza[chosen].mean()), 2),
        "sun_azimuth_deg": round(circular_mean_deg(saa[chosen]), 2),
        "view_zenith_deg": round(float(vza[chosen].mean()), 2),
        "view_azimuth_deg": round(circular_mean_deg(vaa[chosen]), 2),
    }
    print("  geometry (mean over the chosen pixels):", geometry)
    print("  spread of the geometry: sun zenith +-%.1f, view zenith +-%.1f deg" % (sza[chosen].std(), vza[chosen].std()))
    print("  observed TOA reflectance (median; divided by cos sun zenith):", observed)
    print("  spread over pixels:", spread)

    with open(here / TEMPLATE, encoding="utf-8") as f:
        scenario = json.load(f)
    scenario.update(geometry)
    ancillary, ancillary_notes = read_ancillary()
    scenario.update(ancillary)
    print("  from MOD04_L2 / MOD07_L2 (median in the box):", ancillary)
    scenario["name"] = path.name.split(".")[0] + "_" + ".".join(path.name.split(".")[1:3])
    scenario["_note"] = (
        f"Geometry and observed reflectance extracted from {path.name} (box {BOX}); aerosol, ozone, water vapour and "
        "pressure from the MOD04_L2 / MOD07_L2 files named in the extractor (medians in the box). The wind speed is "
        "still the template's: take it from MERRA-2 / ERA5."
    )
    scenario["_ancillary_spread"] = ancillary_notes
    scenario["observed_reflectance"] = observed
    scenario["_observed_spread"] = spread
    out = here / OUTPUT_DIR / f"clear_ocean_{scenario['name']}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(scenario, f, indent=2)
    print(f"\nWrote {out}\nNow set SCENARIO_FILE in run_clear_ocean.py to it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
