"""Cloud-free ocean + atmosphere simulation, to compare with a real satellite pixel.

Simulates the top-of-atmosphere (TOA) reflectance of a cloud-free scene - a
molecular atmosphere (optionally with aerosol) over a Cox-Munk ocean - for the
sun and view geometry of one real observation (e.g. a MODIS Terra pixel), in the
MODIS band set, and prints it next to the observed values if you give them.

Describe the observation in a scenario file (SCENARIO_FILE below; see
configs/clear_ocean_template.json) and run this file:

    python run_clear_ocean.py

Angles are the ones MODIS gives (MOD03: SolarZenith, SolarAzimuth,
SensorZenith, SensorAzimuth): azimuths are measured clockwise from north, of the
direction *toward* the sun / the sensor, as seen from the ground pixel. Result
files go to OUTPUT_DIR.

Uses a plane-parallel atmosphere, one SMART-G run for the geometry; it has no
cloud and needs no 3D grid. Needs a CUDA GPU, like the other scripts.
"""

import json
import sys
from pathlib import Path

import numpy as np

# ============================== SETTINGS ==============================
# A relative path is taken relative to this file's folder.
SCENARIO_FILE = "configs/clear_ocean_MOD021KM_A2023064.1750.json"
OUTPUT_DIR = "clear_ocean_runs"
# Photons of the one run. The scene is horizontally uniform, so the noise of the
# result only depends on this number and on the angular bin the view falls in
# (below), which gets a small share of the photons. With the coarse 15 x 30 bins
# and 1e8 photons repeated runs agreed to about 1% (reflection-only sea: 5 s;
# with the water body: about 250 s on an RTX 3090). With fine bins (90 x 180) and
# 2e8 photons the scatter was about 20%.
N_PHOTONS = 1e8
# Angular output bins (zenith, azimuth) of SMART-G. Coarse bins put more photons in
# each, and are fine where the reflectance changes slowly with angle; near the sun
# glint use finer bins and more photons (the 15 x 30 result matched a 45 x 90 run
# with 2e9 photons to within 1% for the scene in CLEAR_OCEAN_COMPARISON.md).
N_THETA = 15
N_PHI = 30
# =======================================================================

here = Path(__file__).parent

# Same no-display setup as the other scripts (Traits backend, SMARTG_DIR_AUXDATA).
import run_from_config  # noqa: E402

playground_core = run_from_config.playground_core
main = playground_core.main

# MODIS channel names used by the scenario's "observed" values, in the order of
# the "modis" band set (blue, green, red).
MODIS_NAMES = ("b03", "b04", "b01")


def smartg_azimuth_deg(azimuth_from_north_clockwise_deg: float) -> float:
    """MODIS azimuth (clockwise from north) -> SMART-G's local frame (counter-
    clockwise from east), the one main.py uses for the sun and the sensor."""
    return (90.0 - azimuth_from_north_clockwise_deg) % 360.0


def load_scenario(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        scenario = json.load(f)
    required = ("sun_zenith_deg", "sun_azimuth_deg", "view_zenith_deg", "view_azimuth_deg", "latitude_deg")
    missing = [key for key in required if scenario.get(key) is None]
    if missing:
        sys.exit(f"{path}: please set {', '.join(missing)}")
    if not 0.0 <= scenario["sun_zenith_deg"] < 90.0:
        sys.exit(f"{path}: sun_zenith_deg must be in [0, 90)")
    if not 0.0 <= scenario["view_zenith_deg"] < 90.0:
        sys.exit(f"{path}: view_zenith_deg must be in [0, 90)")
    return scenario


def environment_of(scenario: dict):
    """The scenario's aerosol, gas amounts and water body, as playground_core.Environment."""
    return playground_core.Environment(
        aerosol_model=scenario.get("aerosol_model", "maritime_clean") if (scenario.get("aod_550") or 0) > 0 else playground_core.AEROSOL_NONE,
        aod_550=scenario.get("aod_550") or 0.0,
        ozone_dobson=scenario.get("ozone_dobson"),
        water_vapour_g_cm2=scenario.get("water_vapour_g_cm2"),
        surface_pressure_hpa=scenario.get("surface_pressure_hpa"),
        water_body=scenario.get("chlorophyll_mg_m3") is not None,
        chlorophyll_mg_m3=scenario.get("chlorophyll_mg_m3") or playground_core.DEFAULT_CHLOROPHYLL_MG_M3,
        atmosphere_profile=scenario.get("atmosphere_profile"),
    )


def simulate(scenario: dict) -> tuple[tuple[str, ...], np.ndarray]:
    """TOA reflectance in the three MODIS channels (blue, green, red) for the scenario."""
    main.ensure_cuda_compatible_msvc_on_path()
    from smartg.smartg import Smartg

    spectrum = playground_core.make_spectrum(playground_core.BAND_SET_MODIS, playground_core.CHANNEL_ALL)
    environment = environment_of(scenario)
    wind = float(scenario.get("wind_speed_ms", playground_core.DEFAULT_WIND_SPEED_MS))
    surface, water = playground_core.build_ocean(playground_core.SURFACE_OCEAN, wind, environment)
    ds = Smartg().run(
        wavelength=spectrum.run_wavelength,
        atmosphere=playground_core.build_atmosphere_1d(scenario["latitude_deg"], environment),
        surface=surface,
        water=water,
        th_deg=scenario["sun_zenith_deg"],
        ph_deg=smartg_azimuth_deg(scenario["sun_azimuth_deg"]),
        n_photons=N_PHOTONS,
        n_theta=N_THETA,
        n_phi=N_PHI,
        progress=False,
    )
    toa = main._wrap_toa_azimuth(ds["I_up (TOA)"])
    at_view = main.toa_reflectance_at_angles(
        toa,
        np.array([scenario["view_zenith_deg"]]),
        np.array([smartg_azimuth_deg(scenario["view_azimuth_deg"])]),
    )
    return spectrum.labels, spectrum.channels(at_view.reshape(1, -1))[0]


def main_script() -> int:
    path = here / SCENARIO_FILE  # an absolute SCENARIO_FILE wins
    scenario = load_scenario(path)
    labels, simulated = simulate(scenario)

    observed = scenario.get("observed_reflectance") or {}
    print(f"\nScenario {scenario.get('name', path.stem)}: sun zenith {scenario['sun_zenith_deg']:g}, "
          f"view zenith {scenario['view_zenith_deg']:g}, wind {scenario.get('wind_speed_ms', 'default')} m/s, "
          f"AOD(550) {scenario.get('aod_550') or 0:g}")
    print(f"{'channel':28s} {'simulated':>10s} {'observed':>10s} {'sim/obs':>8s}")
    rows = {}
    for name, label, value in zip(MODIS_NAMES, labels, simulated):
        seen = observed.get(name)
        ratio = f"{value / seen:8.3f}" if seen else f"{'':>8s}"
        print(f"{name + ' ' + label:28s} {value:10.4f} {seen if seen is not None else '':>10} {ratio}")
        rows[name] = {"label": label, "simulated": float(value), "observed": seen}

    out = here / OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    result_file = out / f"{path.stem}_result.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump({"scenario": scenario, "n_photons": N_PHOTONS, "channels": rows}, f, indent=2)
    print(f"\nWrote {result_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main_script())
