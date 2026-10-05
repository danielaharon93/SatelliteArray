import argparse
import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from mayavi import mlab
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor
from PIL import Image, ImageOps
from pyface.api import GUI
from pyface.timer.api import Timer
from skyfield.api import EarthSatellite, load, wgs84
from skyfield.framelib import itrs
from traits.api import Button, Enum, HasTraits, Instance, List, Str, observe
from traitsui.api import CheckListEditor, HGroup, Item, VGroup, View
from tvtk.api import tvtk

CELESTRAK_GP_URL = "https://celestrak.org/NORAD/elements/gp.php"
CLOUDCT_SAT_NAME = "CLOUDCT"
EARTH_RADIUS_KM = 6371.0
ORBIT_SAMPLES = 300
ANIMATION_DELAY_MS = 30
# Real-world seconds it should take the animation to complete one simulated
# orbit at the default speed; the satellite's actual orbital period is
# scaled to fit this to pick a sensible starting speed multiplier.
ORBIT_ANIMATION_SECONDS = 60.0
MIN_SPEED_MULTIPLIER = 1.0
MAX_SPEED_MULTIPLIER = 1_000_000.0
SPEED_STEP_FACTOR = 2.0

# The formation: all satellites fly the same orbit, trailing one another
# along-track by a fixed spacing. Satellite 0's position comes straight from
# the TLE; satellites 1..N-1 are positioned relative to it.
NUM_SATELLITES = 10
SATELLITE_SPACING_KM = 100.0
# Smaller than SATELLITE_SPACING_KM so neighbours render as distinct dots
# instead of a fused chain of overlapping spheres.
SATELLITE_MARKER_SCALE_KM = 40.0
# The "5th satellite" (0-indexed) whose nadir point the whole formation
# points at.
NADIR_REFERENCE_INDEX = 4

# Each satellite's imaging sensor is modeled as a rectilinear (pinhole)
# camera with these half-angles: FOV_HALF_ANGLE_X_DEG along the formation's
# cross-track ("right") axis, FOV_HALF_ANGLE_Y_DEG along its along-track /
# orbital-normal ("up") axis.
FOV_HALF_ANGLE_X_DEG = 5.773
FOV_HALF_ANGLE_Y_DEG = 4.235
# Resolution of the sampled FOV-footprint patch drawn on Earth's surface.
FOOTPRINT_GRID_SIZE = 25
# Small radial nudge so the footprint patch doesn't z-fight with Earth's
# own surface, which sits at exactly the same radius.
FOOTPRINT_SURFACE_OFFSET_KM = 3.0

# NASA Blue Marble (public domain) equirectangular Earth texture, cached locally.
EARTH_TEXTURE_URL = (
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/57000/57735/"
    "land_ocean_ice_cloud_2048.jpg"
)
EARTH_TEXTURE_PATH = Path(__file__).with_name("earth_texture.jpg")

# SMART-G auxiliary datasets (aerosols, clouds, gas absorption, etc.), cached
# locally alongside main.py, same pattern as the Earth texture and TLE cache.
SMARTG_AUXDATA_DIR = Path(__file__).with_name("smartg_auxdata")

# SMART-G scene setup for rendering each satellite's acquired image via
# radiative transfer. RGB-ish "true color" bands (blue/green/red), matching
# the wavelengths commonly used for true-color satellite composites.
SMARTG_WAVELENGTHS_NM = [440.0, 550.0, 670.0]

# AFGL standard atmosphere profile (auxdata data_type='atm'); a fixed
# default here rather than one picked from the ground target's latitude
# per render, for a first working pipeline - see Atm1D's `fname` doc for
# the other available profiles (tropical, sub-arctic, ...).
SMARTG_AFGL_PROFILE = "afglus"

# Uniform synthetic water-cloud layer (auxdata data_type='cld'), standing
# in for a real cloud field until/unless this app gets one - see the
# "Cloud source" decision in the project's SMART-G integration notes.
SMARTG_CLOUD_FNAME = "wc"  # water-cloud droplet model
SMARTG_CLOUD_EFFECTIVE_RADIUS_UM = 12.0
SMARTG_CLOUD_BASE_ALTITUDE_KM = 1.0
SMARTG_CLOUD_TOP_ALTITUDE_KM = 2.0
SMARTG_CLOUD_OPTICAL_THICKNESS = 10.0
SMARTG_CLOUD_OPTICAL_THICKNESS_WAVELENGTH_NM = 550.0

# Flat Lambertian surface (dark, ocean-like default) beneath the cloud layer.
SMARTG_SURFACE_ALBEDO = 0.05

# SMARTG_WAVELENGTHS_NM is [blue, green, red]; this indexes it into (R, G, B)
# order for building a displayable image.
SMARTG_RGB_CHANNEL_ORDER = (2, 1, 0)

# Default photon count for a rendering run: a compromise between noise and
# runtime. SMART-G's own run() default (1e9) is tuned for point-geometry
# science use, not a full per-pixel image render.
SMARTG_RENDER_N_PHOTONS = 2e7
# Angular resolution of the single TOA reflectance simulation every
# satellite's image is resampled from - see render_formation_images().
SMARTG_RENDER_N_THETA = 90
SMARTG_RENDER_N_PHI = 180

SMARTG_IMAGES_DIR = Path(__file__).with_name("smartg_images")

# --- 3D scene setup (real cloud spatial structure, --render-mode 3d) ---
# A real LES-simulated cumulus field (I3RC/IPRT benchmark "Case 4"), from
# the 'IPRT' SMART-G auxdata (see ensure_smartg_auxdata(data_type='IPRT')):
# 100x100x53 cells, a 6.67 x 6.67 km horizontal domain, 0-30 km altitude.
SMARTG_3D_CLOUD_PATH = SMARTG_AUXDATA_DIR / "IPRT" / "phaseB" / "grids" / "cumulus.dat"
# The file's own reference wavelength (nm) its extinction values are given
# at (see read_i3rc_cloud()) - unlike the uniform layer's cloud, this can't
# be chosen freely, since it's baked into the file's raw data.
SMARTG_3D_CLOUD_W_REF_NM = 670.0
# The bulk 'wc' cloud optical-property file only supports effective radii
# in [5, 30] um; the raw field has a few cells slightly below 5, clamped
# here rather than left to raise - the same fix SMART-G's own IPRT Phase B
# reference implementation (smartg.iprt.phase_b.build_cloud_c3) applies.
SMARTG_3D_CLOUD_REFF_MIN_UM = 5.0
SMARTG_3D_CLOUD_REFF_ACC = 1

# Non-periodic: the domain's horizontal (x, y) boundary is a hard edge
# rather than wrapping infinitely - see build_smartg_scene_3d(). An extra
# boundary margin (km, added on every side) gives photons that leave the
# native field a defined empty region to travel through before they're
# lost, rather than exiting right at the field's own edge.
#
# Confirmed by a direct comparison: turning periodicity off cuts the
# signal by roughly 30-50x for an oblique-viewing satellite (most of that
# run's signal came from multiply-scattered paths long enough to wrap
# around the domain) and removes the "grid" pattern periodic wrapping was
# producing - see SMARTG_3D_DEFAULT_GAIN.
SMARTG_3D_HORIZ_EXTEND_KM = 20.0

# Per-satellite image resolution, when not given explicitly: each image
# covers the field's *entire* horizontal domain (~6.67 km), not a
# physically-scaled FOV footprint - see render_formation_images_3d() for
# why. Kept low: runtime and GPU memory scale with the pixel count (one
# Sensor per pixel), unlike the uniform layer's single shared simulation.
SMARTG_3D_GRID_SIZE = 20

# Photon budget per satellite image, scaled by its pixel count so noise
# stays roughly constant across resolutions (rather than a flat photon
# count spread over a growing or shrinking number of pixels). Tuned
# against a real run at 20x20: noticeably noisy already at this level: an
# actual "clean" image likely wants several times more, at the cost of
# runtime - see SMARTG_3D_RESOLUTION_PRESETS for how fast this compounds.
SMARTG_3D_PHOTONS_PER_PIXEL = 2000.0

# Tiny offset below the field's own top altitude, keeping sensors placed
# there just inside the domain instead of exactly on its boundary.
SMARTG_3D_SENSOR_TOP_OFFSET_KM = 1e-6

# Image resolution choices for the interactive "Acquire Images" GUI
# control. 4096x3000 is a real acquisition resolution, but - one Sensor
# object per pixel, in current SMART-G - is impractical to actually run at
# that count (tens of minutes of pure-Python sensor-list setup alone per
# satellite, and GPU output buffers scaling with pixel count reaching
# multi-GB per satellite): kept selectable for when that cost is
# acceptable, default is far lower.
SMARTG_3D_RESOLUTION_PRESETS = {
    "Low (64x48)": (64, 48),
    "Medium (256x192)": (256, 192),
    "High (1024x768)": (1024, 768),
    "Full (4096x3000)": (4096, 3000),
}
SMARTG_3D_DEFAULT_RESOLUTION = "Low (64x48)"

# Default brightness gain for saved 3D-mode images (see
# reflectance_to_image()), higher than the uniform layer's (3.0): now that
# the domain is non-periodic (SMARTG_3D_HORIZ_EXTEND_KM), reflectance runs
# noticeably dimmer, especially for oblique-viewing satellites. A single
# fixed gain can't be exactly right for every satellite at once (nadir-ish
# views dim far less than oblique ones), this is a reasonable middle
# ground - override with --render-gain if a given satellite looks over-
# or under-exposed.
SMARTG_3D_DEFAULT_GAIN = 25.0

# Scattering-order check (estimate_scattering_order_fractions()): a smaller,
# faster grid than a real render, since it's a diagnostic, not an image.
SMARTG_SCATTERING_ORDER_GRID_SIZE = 16
# Matches Smartg.run()'s own s_max default - effectively "no upper limit"
# on the number of interactions.
SMARTG_SCATTERING_ORDER_MAX_INTERACTIONS = 1_000_000

# CelesTrak's own usage policy: orbital data is only refreshed server-side
# every 2 hours, and re-requesting more often than that risks a temporary
# IP block ("excessive downloads"). Cache the TLE locally and only refetch
# once it's older than this.
TLE_CACHE_PATH = Path(__file__).with_name("tle_cache.txt")
TLE_CACHE_MAX_AGE_SECONDS = 2 * 3600


def fetch_tle(name: str = CLOUDCT_SAT_NAME) -> str:
    """Request the current TLE for the given satellite name from CelesTrak."""
    response = requests.get(
        CELESTRAK_GP_URL,
        params={"NAME": name, "FORMAT": "TLE"},
        timeout=10,
    )
    response.raise_for_status()

    # CelesTrak sends CRLF line endings; normalize to plain "\n" so caching
    # it with path.write_text() doesn't double them up into "\r\r\n"
    # (Windows text-mode writes translate "\n" -> "\r\n" on top of whatever
    # line endings are already in the string).
    tle = response.text.strip().replace("\r\n", "\n").replace("\r", "\n")
    if not tle:
        raise ValueError(f"No TLE data found for satellite name '{name}'")

    return tle


def get_tle(
    name: str = CLOUDCT_SAT_NAME, path: Path = TLE_CACHE_PATH
) -> str:
    """Return a TLE for `name`, preferring a fresh local cache over hitting
    CelesTrak. Falls back to a stale cache if a live fetch fails (e.g. rate
    limited), so the app keeps working - it just uses slightly older
    orbital elements - rather than hard-failing."""
    if path.exists():
        age_seconds = time.time() - path.stat().st_mtime
        if age_seconds < TLE_CACHE_MAX_AGE_SECONDS:
            return path.read_text().strip()

    try:
        tle = fetch_tle(name)
        path.write_text(tle)
        return tle
    except requests.RequestException as exc:
        if path.exists():
            print(
                f"Live TLE fetch failed ({exc}); using cached TLE from "
                f"{(time.time() - path.stat().st_mtime) / 60:.0f} min ago instead.",
                file=sys.stderr,
            )
            return path.read_text().strip()
        raise


def load_ephemeris():
    """Load the JPL DE421 planetary ephemeris, used to compute the Sun's
    position. Downloaded once (~17MB) and cached as de421.bsp alongside
    main.py, the same pattern as the Earth texture and TLE cache."""
    return load("de421.bsp")


def sun_position_km(eph, t) -> np.ndarray:
    """The Sun's position (km) at time t, in the same Earth-fixed ITRS
    frame used for every other position in this app."""
    earth_to_sun = (eph["sun"] - eph["earth"]).at(t)
    return earth_to_sun.frame_xyz(itrs).km


def build_satellite(tle_text: str) -> EarthSatellite:
    """Parse raw TLE text (2 or 3 line form) into a Skyfield EarthSatellite."""
    lines = tle_text.strip().splitlines()
    if len(lines) == 3:
        name, line1, line2 = (line.strip() for line in lines)
    elif len(lines) == 2:
        name = CLOUDCT_SAT_NAME
        line1, line2 = (line.strip() for line in lines)
    else:
        raise ValueError(f"Unexpected TLE format:\n{tle_text}")

    ts = load.timescale()
    return EarthSatellite(line1, line2, name, ts)


def orbital_period_days(satellite: EarthSatellite) -> float:
    """Derive the orbital period (days) from the TLE's mean motion."""
    revolutions_per_day = satellite.model.no_kozai * 1440.0 / (2 * np.pi)
    return 1.0 / revolutions_per_day


def orbit_plane_loop_km(
    satellite: EarthSatellite, t, num_points: int = ORBIT_SAMPLES
) -> np.ndarray:
    """A closed loop of points (num_points, 3), in Earth-fixed ITRS km,
    tracing the satellite's *current* instantaneous orbital plane as a
    circle at its current orbital radius.

    The orbital plane's normal (specific angular momentum, r x v) is
    computed in the inertial GCRS frame, where it stays fixed for
    unperturbed motion - that's what makes the loop always closed, unlike a
    ground track. The resulting circle is then rotated into ITRS via the
    same rotation Skyfield uses for frame_xyz(itrs), so the loop visibly
    turns beneath the fixed loop as Earth rotates: it always shows where
    the orbital plane currently slices through Earth's surface."""
    geocentric = satellite.at(t)
    r_gcrs = geocentric.position.km
    v_gcrs = geocentric.velocity.km_per_s

    radius = np.linalg.norm(r_gcrs)
    normal = np.cross(r_gcrs, v_gcrs)
    normal /= np.linalg.norm(normal)

    u1 = r_gcrs / radius
    u2 = np.cross(normal, u1)

    theta = np.linspace(0.0, 2 * np.pi, num_points)
    circle_gcrs = radius * (
        np.outer(np.cos(theta), u1) + np.outer(np.sin(theta), u2)
    )

    rotation = itrs.rotation_at(t)
    return circle_gcrs @ rotation.T


def orbital_speed_km_per_s(satellite: EarthSatellite, t) -> float:
    """Instantaneous orbital speed (km/s) at time t."""
    return float(np.linalg.norm(satellite.at(t).velocity.km_per_s))


def formation_time_offsets_seconds(
    satellite: EarthSatellite,
    t0,
    num_satellites: int = NUM_SATELLITES,
    spacing_km: float = SATELLITE_SPACING_KM,
) -> np.ndarray:
    """Along-track time lag (seconds) for each satellite in the formation,
    trailing satellite 0 (the TLE's own position). Computed once from the
    orbital speed at t0: since every satellite flies the identical orbit
    just time-shifted, a fixed time lag keeps a ~fixed arc-length spacing
    between neighbours (exact for a circular orbit; this orbit's
    eccentricity is ~0.0008, so the approximation is essentially exact)."""
    dt_seconds = spacing_km / orbital_speed_km_per_s(satellite, t0)
    return dt_seconds * np.arange(num_satellites)


def formation_positions_km(
    satellite: EarthSatellite, t, time_offsets_seconds: np.ndarray
) -> np.ndarray:
    """Earth-fixed (ITRS) positions (num_satellites, 3) km for the whole
    formation, all *simultaneously* at time t: satellite i's inertial
    (GCRS) position equals satellite 0's own position `time_offsets_seconds`
    earlier - since every satellite retraces the identical orbital path,
    that's exactly where a trailing satellite sits right now.

    Crucially, that per-satellite GCRS position must then be converted to
    ITRS using Earth's rotation state at the single shared time t, not at
    each satellite's own shifted time - all satellites exist at the same
    real moment, against the same real, currently-rotating Earth. Using the
    per-satellite rotation instead (frame_xyz(itrs) on the shifted times)
    silently inflates the along-track spacing (~100km came out ~101km when
    tried the naive way)."""
    times = t + (-time_offsets_seconds / 86400.0)
    positions_gcrs = satellite.at(times).position.km  # shape (3, N)
    rotation = itrs.rotation_at(t)
    return positions_gcrs.T @ rotation.T


def nadir_target_km(position_km: np.ndarray) -> np.ndarray:
    """The point on Earth's (spherical) surface where the straight line
    through this position and Earth's center crosses - i.e. this position's
    nadir / sub-satellite point."""
    return EARTH_RADIUS_KM * position_km / np.linalg.norm(position_km)


def formation_pointing_vectors_km(
    positions_km: np.ndarray, reference_index: int = NADIR_REFERENCE_INDEX
) -> np.ndarray:
    """The vector (km, from each satellite toward the shared ground target)
    every satellite in the formation points along: the nadir point of
    `reference_index` (the "5th satellite"). That satellite's own vector
    comes out to nadir too, since the target is derived from its own
    position - satellite, target and Earth's center are collinear for it,
    same as the geometry the task describes."""
    target = nadir_target_km(positions_km[reference_index])
    return target - positions_km  # shape (N, 3), one row per satellite


def enu_basis_km(position_km: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The (east, north, up) unit vectors of the local East-North-Up frame
    at an Earth-fixed (ITRS) position - the reference frame SMART-G's own
    geometry convention (sun angles, and each Sensor's viewing angles) is
    defined in, see zenith_azimuth_deg().

    `up` is the local vertical: the position's own direction from Earth's
    center, exact for a sphere - the same spherical-Earth approximation
    `nadir_target_km()` uses. `east`/`north` complete a right-handed frame;
    like any ENU frame this degenerates at the geographic poles (`east` is
    undefined there), never reached by any ground target this app uses."""
    up = position_km / np.linalg.norm(position_km)
    north_pole = np.array([0.0, 0.0, 1.0])
    east = np.cross(north_pole, up)
    east /= np.linalg.norm(east)
    north = np.cross(up, east)
    return east, north, up


def zenith_azimuth_deg(direction_km: np.ndarray, origin_km: np.ndarray) -> tuple[float, float]:
    """The (zenith, azimuth) angles in degrees of `direction_km` (a vector
    from `origin_km` toward whatever is being measured - a satellite
    position, the Sun's position, ...; only its direction matters, not its
    length), in SMART-G's own local-frame convention (matches
    geoclide.vec2ang with vec_view='zenith', which is what a SMART-G
    `Sensor`'s default th_deg=0/ph_deg direction and the sun angles passed
    to `Smartg.run()` are defined against): zenith is measured from
    straight up (0 deg) at `origin_km`, and azimuth counterclockwise from
    local east (matching vec2ang's "from x+, trigonometric around z")."""
    east, north, up = enu_basis_km(origin_km)
    direction = direction_km / np.linalg.norm(direction_km)

    x = np.dot(direction, east)
    y = np.dot(direction, north)
    z = np.dot(direction, up)

    zenith_deg = np.degrees(np.arccos(np.clip(z, -1.0, 1.0)))
    azimuth_deg = np.degrees(np.arctan2(y, x)) % 360.0
    return float(zenith_deg), float(azimuth_deg)


def solar_angles_deg(sun_position_km: np.ndarray, target_km: np.ndarray) -> tuple[float, float]:
    """Solar zenith and azimuth angles (degrees), i.e. SZA/SAA, of the Sun
    as seen from `target_km` (e.g. a satellite's nadir point) - the
    `th_deg`/`ph_deg` SMART-G's `Smartg.run()` expects for the sun
    direction, in its local-frame convention (see zenith_azimuth_deg())."""
    return zenith_azimuth_deg(sun_position_km - target_km, target_km)


def viewing_angles_deg(
    satellite_position_km: np.ndarray, target_km: np.ndarray
) -> tuple[float, float]:
    """Viewing zenith and azimuth angles (degrees), i.e. VZA/VAA, of a
    satellite as seen from `target_km` on the ground - the `th_deg`/`ph_deg`
    a SMART-G `Sensor` placed at `target_km` and looking up toward that
    satellite would use for backward-mode imaging, in SMART-G's local-frame
    convention (see zenith_azimuth_deg())."""
    return zenith_azimuth_deg(satellite_position_km - target_km, target_km)


def orbital_normal_itrs(satellite: EarthSatellite, t) -> np.ndarray:
    """Unit vector normal to the orbital plane (specific angular momentum
    direction, r x v), rotated into the Earth-fixed ITRS frame at time t.
    Used as a stable "up" direction for the nadir-following camera, since
    it stays perpendicular to the formation's orbital plane."""
    geocentric = satellite.at(t)
    r_gcrs = geocentric.position.km
    v_gcrs = geocentric.velocity.km_per_s
    normal_gcrs = np.cross(r_gcrs, v_gcrs)
    normal_gcrs /= np.linalg.norm(normal_gcrs)
    return itrs.rotation_at(t) @ normal_gcrs


def nadir_camera_frame_km(reference_position_km: np.ndarray, view_up_km: np.ndarray):
    """Camera (position, focal_point, view_up) placed exactly at satellite
    4's own position, looking straight down its nadir line at the ground
    point beneath it - i.e. literally what that satellite sees, not an
    outside view of the formation. Recomputed from that satellite's live
    position every tick, so the view continuously turns to track wherever
    its nadir points as the formation orbits."""
    return reference_position_km, nadir_target_km(reference_position_km), view_up_km


def ray_sphere_intersection_km(
    position_km: np.ndarray,
    directions_unit: np.ndarray,
    radius_km: float = EARTH_RADIUS_KM,
) -> np.ndarray:
    """Nearest intersection of rays from `position_km` (3,), each along a
    unit direction in `directions_unit` (..., 3), with a sphere of
    `radius_km` centered on the origin. Vectorized over any leading shape
    of `directions_unit`. Discriminant is clipped at 0 (grazing hit)
    instead of going negative, so this never returns NaN even for a
    direction that technically misses the sphere - harmless here since
    every ray used in this app is within a few degrees of a boresight that
    is already known to hit Earth."""
    b = np.tensordot(directions_unit, position_km, axes=([-1], [0]))
    c = np.dot(position_km, position_km) - radius_km**2
    discriminant = np.clip(b * b - c, 0.0, None)
    t = -b - np.sqrt(discriminant)
    return position_km + t[..., None] * directions_unit


def fov_footprint_km(
    position_km: np.ndarray,
    target_km: np.ndarray,
    up_reference_km: np.ndarray,
    half_angle_x_deg: float = FOV_HALF_ANGLE_X_DEG,
    half_angle_y_deg: float = FOV_HALF_ANGLE_Y_DEG,
    grid_size: int = FOOTPRINT_GRID_SIZE,
) -> np.ndarray:
    """The ground footprint (grid_size, grid_size, 3) km that a satellite's
    rectilinear-camera FOV covers on Earth's surface: a
    2*half_angle_x_deg x 2*half_angle_y_deg cone of view rays from
    `position_km` toward `target_km`, intersected with Earth's sphere and
    nudged slightly above it (FOOTPRINT_SURFACE_OFFSET_KM) so it renders
    cleanly over the Earth texture instead of z-fighting with it.

    `up_reference_km` (the shared orbital-plane normal) sets the camera's
    roll: it's projected perpendicular to the boresight to build an
    orthonormal (boresight, right, up) frame, which works even for
    satellites that aren't looking straight down (everyone except the
    reference satellite itself, who are squinting sideways at the shared
    target)."""
    boresight = target_km - position_km
    boresight = boresight / np.linalg.norm(boresight)

    up = up_reference_km - np.dot(up_reference_km, boresight) * boresight
    up = up / np.linalg.norm(up)
    right = np.cross(boresight, up)

    tx = np.tan(np.radians(np.linspace(-half_angle_x_deg, half_angle_x_deg, grid_size)))
    ty = np.tan(np.radians(np.linspace(-half_angle_y_deg, half_angle_y_deg, grid_size)))
    tx_grid, ty_grid = np.meshgrid(tx, ty)

    directions = boresight + tx_grid[:, :, None] * right + ty_grid[:, :, None] * up
    directions /= np.linalg.norm(directions, axis=2, keepdims=True)

    footprint = ray_sphere_intersection_km(position_km, directions)
    return footprint * ((EARTH_RADIUS_KM + FOOTPRINT_SURFACE_OFFSET_KM) / EARTH_RADIUS_KM)


def print_current_state(satellite: EarthSatellite) -> None:
    ts = load.timescale()
    now = ts.now()
    geocentric = satellite.at(now)
    subpoint = wgs84.subpoint(geocentric)
    x, y, z = geocentric.frame_xyz(itrs).km

    print(f"Satellite: {satellite.name}")
    print(f"Time (UTC): {now.utc_strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"ITRS (Earth-fixed) position (km): x={x:.1f}, y={y:.1f}, z={z:.1f}")
    print(
        f"Subpoint: lat={subpoint.latitude.degrees:.3f}deg, "
        f"lon={subpoint.longitude.degrees:.3f}deg, "
        f"alt={subpoint.elevation.km:.1f}km"
    )
    print(f"Orbital period: {orbital_period_days(satellite) * 1440.0:.1f} min")
    print(
        f"Formation: {NUM_SATELLITES} satellites, "
        f"{SATELLITE_SPACING_KM:.0f} km apart along-track"
    )


def ensure_earth_texture(path: Path = EARTH_TEXTURE_PATH) -> Path | None:
    """Download and cache the Earth texture image if not already present.
    Returns None (instead of raising) if it can't be obtained, so the caller
    can fall back to a plain sphere.

    The image is mirrored horizontally on download: VTK's TexturedSphereSource
    parametrizes longitude with the opposite handedness of geographic
    (east-positive) longitude, so an unmirrored equirectangular map ends up
    east-west flipped once wrapped onto the sphere. Combined with the
    actor.rotate_y(180) applied in add_earth(), this makes the sphere's
    surface line up exactly with ITRS (Earth-fixed) coordinates - verified
    numerically to sub-degree accuracy against the raw VTK mesh geometry."""
    if path.exists():
        return path

    try:
        response = requests.get(EARTH_TEXTURE_URL, timeout=15)
        response.raise_for_status()
        mirrored = ImageOps.mirror(Image.open(BytesIO(response.content)))
        mirrored.save(path, format="JPEG", quality=90)
    except requests.RequestException as exc:
        print(f"Could not download Earth texture ({exc}); using plain sphere.", file=sys.stderr)
        return None

    return path


def ensure_smartg_auxdata(
    path: Path = SMARTG_AUXDATA_DIR, data_type: str | list[str] = "all"
) -> None:
    """Download SMART-G's auxiliary datasets (aerosols, clouds, gas
    absorption tables, etc.) into `path` if not already present there.
    Files already on disk are skipped automatically by `smartg.auxdata`, so
    this is safe to call on every run. Non-fatal on failure (e.g. offline),
    matching the fallback pattern used by ensure_earth_texture()."""
    from smartg.auxdata import download

    try:
        path.mkdir(parents=True, exist_ok=True)
        download(path, data_type=data_type)
    except Exception as exc:  # smartg.auxdata doesn't document a narrower type
        print(f"Could not download SMART-G auxiliary data ({exc}).", file=sys.stderr)


def build_smartg_scene(lat_deg: float):
    """Build the SMART-G atmosphere + surface scene used to render each
    satellite's acquired image: a standard AFGL atmosphere profile carrying
    a uniform synthetic water-cloud layer, over a flat Lambertian surface.

    `lat_deg` (the ground target's latitude, e.g. from
    `wgs84.subpoint(...).latitude.degrees`) feeds `Atm1D`'s Rayleigh
    optical-depth calculation, which depends on latitude.

    Requires the 'atm' and 'cld' SMART-G auxdata (see
    ensure_smartg_auxdata()) to already be downloaded.

    Returns (wavelength_nm, atmosphere, surface): pass them straight to
    `Smartg().run(wavelength=wavelength_nm, atmosphere=atmosphere,
    surface=surface, ...)`."""
    from smartg.albedo import AlbedoCst
    from smartg.atmosphere import Atm1D, Cloud
    from smartg.surface import LambSurface

    cloud = Cloud(
        SMARTG_CLOUD_FNAME,
        SMARTG_CLOUD_EFFECTIVE_RADIUS_UM,
        SMARTG_CLOUD_BASE_ALTITUDE_KM,
        SMARTG_CLOUD_TOP_ALTITUDE_KM,
        SMARTG_CLOUD_OPTICAL_THICKNESS,
        SMARTG_CLOUD_OPTICAL_THICKNESS_WAVELENGTH_NM,
    )
    atmosphere = Atm1D(SMARTG_AFGL_PROFILE, comp=[cloud], lat=lat_deg)
    surface = LambSurface(alb=AlbedoCst(SMARTG_SURFACE_ALBEDO))

    return np.array(SMARTG_WAVELENGTHS_NM), atmosphere, surface


def build_smartg_scene_3d(lat_deg: float):
    """Build the SMART-G 3D atmosphere + surface scene used to render each
    satellite's acquired image with real cloud spatial structure: a
    standard AFGL background atmosphere carrying the real LES-simulated
    cumulus field at SMARTG_3D_CLOUD_PATH, over a flat Lambertian surface.

    Unlike build_smartg_scene()'s uniform layer, this scene is *not*
    horizontally homogeneous, so a single simulation's result is only
    valid at the specific (x, y) positions it was computed for - see
    render_formation_images_3d(), which can no longer reuse one run
    across the whole formation the way render_formation_images() does.

    The domain's horizontal boundary is non-periodic (a hard edge, with
    an empty extra margin - see SMARTG_3D_HORIZ_EXTEND_KM): a photon path
    that strays past the native 6.67 km field is lost rather than
    wrapping back into it. This trades away some real signal - a
    multiply-scattered path that would have wrapped back in under
    periodic boundaries instead just exits and is lost, so especially for
    oblique-viewing satellites the render comes out dimmer than a
    (perfectly tileable, but visually patterned) periodic domain would
    give - see SMARTG_3D_DEFAULT_GAIN and render_formation_images_3d().

    `lat_deg` feeds `Atm1D`'s Rayleigh optical-depth calculation, same as
    build_smartg_scene().

    Requires the 'atm', 'cld' and 'IPRT' SMART-G auxdata (see
    ensure_smartg_auxdata()) to already be downloaded.

    Returns (wavelength_nm, atmosphere, surface, grid_3d): pass the first
    three straight to `Smartg(opt3d=True).run(wavelength=wavelength_nm,
    atmosphere=atmosphere, surface=surface, ...)`; `grid_3d` is needed to
    place `Sensor`s in the field (see smartg.sensor.get_sensors_grid)."""
    from smartg.albedo import AlbedoCst
    from smartg.atmosphere import Atm1D, Atm3D, Cloud3D, read_i3rc_cloud
    from smartg.grid3d import Grid3D
    from smartg.surface import LambSurface

    if not SMARTG_3D_CLOUD_PATH.exists():
        raise FileNotFoundError(
            f"{SMARTG_3D_CLOUD_PATH} not found; download it with "
            "ensure_smartg_auxdata(data_type='IPRT')."
        )

    field = read_i3rc_cloud(SMARTG_3D_CLOUD_PATH)
    cloud_3d = Cloud3D(
        SMARTG_CLOUD_FNAME,
        w_ref=SMARTG_3D_CLOUD_W_REF_NM,
        ds=field,
        reff_acc=SMARTG_3D_CLOUD_REFF_ACC,
        reff_min=SMARTG_3D_CLOUD_REFF_MIN_UM,
    )
    grid_3d = Grid3D(
        field["x_bounds"].values,
        field["y_bounds"].values,
        field["z_bounds"].values,
        periodic=False,
        horiz_extend_length=SMARTG_3D_HORIZ_EXTEND_KM,
    )
    atm_1d = Atm1D(SMARTG_AFGL_PROFILE, lat=lat_deg)
    atmosphere = Atm3D(atm_1d=atm_1d, grid_3d=grid_3d, comp_3d=[cloud_3d])
    surface = LambSurface(alb=AlbedoCst(SMARTG_SURFACE_ALBEDO))

    return np.array(SMARTG_WAVELENGTHS_NM), atmosphere, surface, grid_3d


def geocentric_latitude_deg(position_km: np.ndarray) -> float:
    """The geocentric latitude (degrees) of an Earth-fixed (ITRS) position,
    on the same spherical-Earth approximation used throughout this app
    (see nadir_target_km())."""
    return float(np.degrees(np.arcsin(position_km[2] / np.linalg.norm(position_km))))


def _wrap_toa_azimuth(toa_reflectance):
    """Append a duplicate azimuth=360deg slice (equal to azimuth=0) to a
    SMART-G `I_up (TOA)` DataArray, so interpolating a pixel whose azimuth
    falls just under 360deg doesn't land outside the coordinate range and
    come back NaN. `Smartg.run()`'s own azimuth bins run from 0deg up to
    (but not including) 360deg (see `cone_sampling`), so this only extends
    the upper edge; no equivalent gap exists at 0deg."""
    import xarray as xr

    edge = toa_reflectance.isel(**{"Azimuth angles": 0})
    edge = edge.assign_coords(**{"Azimuth angles": 360.0})
    return xr.concat([toa_reflectance, edge], dim="Azimuth angles")


def toa_reflectance_at_angles(
    toa_reflectance, zenith_deg: np.ndarray, azimuth_deg: np.ndarray
) -> np.ndarray:
    """Resample a (wrapped) SMART-G `I_up (TOA)` DataArray - dimensionless
    top-of-atmosphere reflectance as a function of viewing (zenith,
    azimuth), plus a `wavelength` dimension if it carries more than one -
    at arbitrary, pixel-wise viewing angles (same-shaped `zenith_deg`/
    `azimuth_deg` arrays, in SMART-G's local-frame convention, see
    viewing_angles_deg()).

    `bounds_error=False, fill_value=None` (scipy's own spelling for "keep
    extrapolating") covers viewing angles that fall just outside the
    simulation's zenith bin centers (its bins span [0, sza_max] but are
    centered mid-bin, see `cone_sampling`) rather than returning NaN pixels
    at the extreme edge of a satellite's field of view.

    Returns an array shaped like `zenith_deg` with an extra trailing
    `wavelength` axis, e.g. (*zenith_deg.shape, n_wavelengths)."""
    import xarray as xr

    zenith = xr.DataArray(zenith_deg, dims=[f"dim{i}" for i in range(zenith_deg.ndim)])
    azimuth = xr.DataArray(azimuth_deg, dims=zenith.dims)
    sampled = toa_reflectance.interp(
        **{"Zenith angles": zenith, "Azimuth angles": azimuth},
        kwargs={"bounds_error": False, "fill_value": None},
    )
    return sampled.transpose(*zenith.dims, "wavelength").values


def _find_vs2022_msvc_bin_dir() -> Path | None:
    """Locate the MSVC host-compiler bin directory from a Visual Studio
    2022 install (Build Tools/Community/Professional/Enterprise, any of
    them), via Microsoft's own `vswhere` tool. Returns None if `vswhere`,
    a matching VS2022 install, or its compiler can't be found.

    Needed because SMART-G compiles a CUDA kernel via pycuda/nvcc the
    first time a `Smartg()` is built, and nvcc only accepts MSVC from
    VS2019 through VS2022 as its host compiler - a newer (or older) cl.exe
    that happens to come first on PATH makes the compile fail with
    "unsupported Microsoft Visual Studio version"."""
    vswhere = (
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
        / "Microsoft Visual Studio"
        / "Installer"
        / "vswhere.exe"
    )
    if not vswhere.exists():
        return None

    try:
        result = subprocess.run(
            [
                str(vswhere),
                "-products", "*",
                "-version", "[17.0,18.0)",  # the VS2022 generation only
                "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property", "installationPath",
                "-latest",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None

    install_path = result.stdout.strip()
    if not install_path:
        return None

    version_file = (
        Path(install_path)
        / "VC" / "Auxiliary" / "Build" / "Microsoft.VCToolsVersion.default.txt"
    )
    try:
        toolset_version = version_file.read_text().strip()
    except OSError:
        return None

    bin_dir = (
        Path(install_path)
        / "VC" / "Tools" / "MSVC" / toolset_version / "bin" / "Hostx64" / "x64"
    )
    return bin_dir if (bin_dir / "cl.exe").exists() else None


def ensure_cuda_compatible_msvc_on_path() -> None:
    """Prepend a VS2022 MSVC bin directory to PATH, if one can be found
    (see _find_vs2022_msvc_bin_dir()) and isn't there already, so pycuda's
    nvcc invocation picks a host compiler it actually supports instead of
    whichever cl.exe happens to come first on PATH.

    Safe to call more than once (a no-op once applied). Also a no-op if no
    VS2022 install is found - the CUDA compile then fails on its own with
    a clearer nvcc error, rather than this function raising."""
    bin_dir = _find_vs2022_msvc_bin_dir()
    if bin_dir is None:
        return

    bin_dir_str = str(bin_dir)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if bin_dir_str in path_entries:
        return

    os.environ["PATH"] = bin_dir_str + os.pathsep + os.environ.get("PATH", "")
    print(f"Using MSVC host compiler for CUDA: {bin_dir_str}", file=sys.stderr)


def render_formation_images(
    satellite: EarthSatellite,
    t,
    time_offsets_seconds: np.ndarray,
    eph,
    n_photons: float = SMARTG_RENDER_N_PHOTONS,
    n_theta: int = SMARTG_RENDER_N_THETA,
    n_phi: int = SMARTG_RENDER_N_PHI,
    grid_size: int = FOOTPRINT_GRID_SIZE,
) -> dict[int, np.ndarray]:
    """Render the image each satellite in the formation acquires at time
    `t`: the top-of-atmosphere reflectance across its FOV footprint, from a
    SMART-G radiative-transfer simulation of the shared scene (see
    build_smartg_scene()).

    A single simulation suffices for the whole formation, run once for the
    solar geometry at the formation's shared nadir target. Its angular
    reflectance output, `I_up (TOA)` as a function of viewing (zenith,
    azimuth), is then resampled once per satellite at that satellite's own
    per-pixel viewing angles (see toa_reflectance_at_angles()). This is
    valid because the scene is horizontally homogeneous (plane-parallel
    atmosphere, uniform cloud layer - see build_smartg_scene()): the same
    TOA angular reflectance applies at every ground point in the
    formation's shared footprint, regardless of which satellite is
    looking, or from where - only the viewing angle differs.

    Requires a CUDA-capable GPU and the SMART-G auxdata already downloaded
    (see ensure_smartg_auxdata()).

    Returns {satellite_index: (grid_size, grid_size, n_wavelengths)
    reflectance array}, wavelengths ordered as SMARTG_WAVELENGTHS_NM."""
    ensure_cuda_compatible_msvc_on_path()
    from smartg.smartg import Smartg

    positions = formation_positions_km(satellite, t, time_offsets_seconds)
    target = nadir_target_km(positions[NADIR_REFERENCE_INDEX])
    view_up = orbital_normal_itrs(satellite, t)

    sza_deg, saa_deg = solar_angles_deg(sun_position_km(eph, t), target)
    if sza_deg >= 90.0:
        raise ValueError(
            f"Sun is below the horizon at the formation's target "
            f"(solar zenith angle={sza_deg:.1f} deg); nothing to render."
        )
    wavelength_nm, atmosphere, surface = build_smartg_scene(
        geocentric_latitude_deg(target)
    )

    ds = Smartg().run(
        wavelength=wavelength_nm,
        atmosphere=atmosphere,
        surface=surface,
        th_deg=sza_deg,
        ph_deg=saa_deg,
        n_photons=n_photons,
        n_theta=n_theta,
        n_phi=n_phi,
        progress=False,
    )
    toa_reflectance = _wrap_toa_azimuth(ds["I_up (TOA)"])

    images = {}
    for index, position in enumerate(positions):
        footprint = fov_footprint_km(position, target, view_up, grid_size=grid_size)
        zenith_deg = np.empty(footprint.shape[:2])
        azimuth_deg = np.empty(footprint.shape[:2])
        for row in range(footprint.shape[0]):
            for col in range(footprint.shape[1]):
                zenith_deg[row, col], azimuth_deg[row, col] = viewing_angles_deg(
                    position, footprint[row, col]
                )
        images[index] = toa_reflectance_at_angles(toa_reflectance, zenith_deg, azimuth_deg)

    return images


def render_formation_images_3d(
    satellite: EarthSatellite,
    t,
    time_offsets_seconds: np.ndarray,
    eph,
    n_photons: float | None = None,
    width_px: int = SMARTG_3D_GRID_SIZE,
    height_px: int | None = None,
    satellite_indices: Sequence[int] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[int, np.ndarray]:
    """Render the image each satellite in the formation acquires at time
    `t`, with real cloud spatial structure, from a SMART-G 3D
    radiative-transfer simulation of the cumulus field (see
    build_smartg_scene_3d()).

    Approach: unlike render_formation_images()'s uniform layer, this scene
    is not horizontally homogeneous, so each satellite needs its own
    simulated view - one `Sensor` grid per satellite
    (`smartg.sensor.get_sensors_grid`, backward mode: each sensor sits at
    the field's top, aimed down along that satellite's own viewing angle,
    and accumulates the radiance reaching it from the sun's direction via
    a local estimate). A sensor's `(th_deg, ph_deg)` is the direction it
    looks *along* (from the sensor toward the ground), the antipode of the
    satellite's own viewing (zenith, azimuth) as seen from the ground (see
    viewing_angles_deg()): `th_deg = 180 - VZA`, `ph_deg = VAA + 180`.

    Satellites are rendered with **one `Smartg.run()` call each**, in a
    Python loop, reusing a single compiled `Smartg` instance (its own
    docstring: built once, cheaply reused across runs) - not batched into
    one combined sensor list, because the GPU's own output buffers scale
    with total sensor count: batching every satellite together could need
    several times a single satellite's GPU memory at once. The atmosphere
    is also only built and `.calc()`-ed once upfront, reused unchanged
    across every satellite's run, since only the sensor geometry differs
    between them - `.calc()` is the expensive step (building the cloud's
    per-cell phase matrices).

    Approximation: each satellite's image covers the field's *entire*
    ~6.67 km horizontal domain, rather than that satellite's true,
    physically-scaled FOV footprint (typically tens of km at orbital
    altitude) - the formation's satellites end up viewing the same patch
    from their different angles, not their true relative ground offsets.
    This keeps the render tractable; see build_smartg_scene_3d() for the
    domain's exact extent.

    The domain's horizontal boundary is non-periodic (see
    build_smartg_scene_3d()): confirmed by a direct comparison, this
    trades away a periodic domain's "grid"-patterned look (real, but tied
    to the domain's own small, repeating size rather than genuine cloud
    structure) for a dimmer, especially at oblique viewing angles, but
    pattern-free result - see SMARTG_3D_DEFAULT_GAIN.

    Parameters
    ----------
    n_photons : photon budget *per satellite image*. None (default) scales
        it with the requested pixel count via SMARTG_3D_PHOTONS_PER_PIXEL,
        so per-pixel noise stays roughly comparable across resolutions.
    width_px, height_px : image resolution. height_px defaults to
        width_px (a square image). Note the field's domain is square but
        this need not be - a non-square resolution stretches it to fit,
        it does not crop or letterbox.
    satellite_indices : which of the formation's NUM_SATELLITES to render
        (0-based). None (default) renders all of them. Rendering fewer is
        the main lever for a faster, cheaper run - each selected
        satellite costs its own full Smartg.run() call.
    progress_callback : if given, called as `progress_callback(done,
        total)` after each satellite's image finishes (`total ==
        len(satellite_indices)`), e.g. to update a GUI status message.

    Requires a CUDA-capable GPU and the SMART-G auxdata already downloaded,
    including 'IPRT' (see ensure_smartg_auxdata()).

    Returns {satellite_index: (height_px, width_px, n_wavelengths)
    reflectance array}, wavelengths ordered as SMARTG_WAVELENGTHS_NM."""
    ensure_cuda_compatible_msvc_on_path()
    from smartg.sensor import get_sensors_grid
    from smartg.smartg import LocalEstimate, Smartg

    if height_px is None:
        height_px = width_px
    if satellite_indices is None:
        satellite_indices = list(range(NUM_SATELLITES))
    if n_photons is None:
        n_photons = SMARTG_3D_PHOTONS_PER_PIXEL * width_px * height_px

    positions = formation_positions_km(satellite, t, time_offsets_seconds)
    target = nadir_target_km(positions[NADIR_REFERENCE_INDEX])

    sza_deg, saa_deg = solar_angles_deg(sun_position_km(eph, t), target)
    if sza_deg >= 90.0:
        raise ValueError(
            f"Sun is below the horizon at the formation's target "
            f"(solar zenith angle={sza_deg:.1f} deg); nothing to render."
        )
    wavelength_nm, atmosphere, surface, grid_3d = build_smartg_scene_3d(
        geocentric_latitude_deg(target)
    )
    # Computed once upfront (see docstring) rather than letting each
    # satellite's run() call redo it via an Atm3D passed straight through.
    atmosphere_profile = atmosphere.calc(wavelength_nm)

    pos_z = grid_3d.zGRID[-1] - SMARTG_3D_SENSOR_TOP_OFFSET_KM
    xgrid = np.linspace(grid_3d.xgrid[0], grid_3d.xgrid[-1], width_px + 1)
    ygrid = np.linspace(grid_3d.ygrid[0], grid_3d.ygrid[-1], height_px + 1)
    cell_size = xgrid[1] - xgrid[0]

    le = LocalEstimate(
        th_deg=np.array([sza_deg]),
        phi_deg=np.array([saa_deg]),
        count_level=np.array([0]),  # 0 = UPTOA, see smartg.sensor.Sensor
    )

    sg = Smartg(back=True, opt3d=True)

    images = {}
    for done, index in enumerate(satellite_indices, start=1):
        vza_deg, vaa_deg = viewing_angles_deg(positions[index], target)
        sensors = get_sensors_grid(
            xgrid,
            ygrid,
            pos_z=pos_z,
            th_deg=180.0 - vza_deg,
            ph_deg=(vaa_deg + 180.0) % 360.0,
            loc="ATMOS",
            cell_size=cell_size,
            grid_3d=grid_3d,
        )
        ds = sg.run(
            wavelength=wavelength_nm,
            atmosphere=atmosphere_profile,
            surface=surface,
            sensor=sensors,
            le=le,
            n_photons=n_photons,
            progress=False,
        )
        # One local-estimate direction (the sun) -> drop those two
        # singleton axes, keeping (sensor index, wavelength).
        toa_reflectance = ds["I_up (TOA)"].isel(
            **{"Azimuth angles": 0, "Zenith angles": 0}
        )
        images[index] = toa_reflectance.values.reshape(
            height_px, width_px, len(wavelength_nm)
        )
        if progress_callback is not None:
            progress_callback(done, len(satellite_indices))

    return images


def estimate_scattering_order_fractions(
    satellite: EarthSatellite,
    t,
    time_offsets_seconds: np.ndarray,
    eph,
    satellite_index: int = NADIR_REFERENCE_INDEX,
    width_px: int = SMARTG_SCATTERING_ORDER_GRID_SIZE,
    height_px: int | None = None,
    n_photons: float | None = None,
) -> dict[str, float]:
    """Check whether photons in the 3D cumulus scene (see
    build_smartg_scene_3d()) tend to reach a given satellite's view after
    scattering once, or multiple times.

    Renders the same view three times - identical scene, sensors, sun
    geometry and random seed - varying only `Smartg.run()`'s `s_min`/
    `s_max`, which filter photons by their total interaction count (the
    same `ph->nint` counter the SMART-G `smartg.histories` module also
    exposes, but that module is built for a different, heavier feature
    -ALIS spectral reconstruction- and needs `Smartg(alis=True)` plus
    `alis_options=Alis(hist=True)`; `s_min`/`s_max` reach the same counter
    directly, with no extra machinery, and work with this scene's existing
    `opt3d=True, back=True` setup):

    - all orders (`s_min=0`, effectively unlimited `s_max`)
    - single-scatter only (`s_min=1, s_max=1`)
    - multi-scatter only (`s_min=2`, effectively unlimited `s_max`)

    Since these are the same fixed scene/photon budget/seed, the "single"
    and "multi" mean reflectances are effectively a breakdown of the "all
    orders" one into its single- vs multiple-scattering contributions
    (they should sum close to it - see the returned 'all_orders_mean').

    Requires a CUDA-capable GPU and the SMART-G auxdata already
    downloaded, including 'IPRT' (see ensure_smartg_auxdata()).

    Returns {'all_orders_mean': float, 'single_scatter_fraction': float,
    'multi_scatter_fraction': float}, the fractions each summing what
    share of the all-orders mean reflectance came from that regime."""
    ensure_cuda_compatible_msvc_on_path()
    from smartg.sensor import get_sensors_grid
    from smartg.smartg import LocalEstimate, Smartg

    if height_px is None:
        height_px = width_px
    if n_photons is None:
        n_photons = SMARTG_3D_PHOTONS_PER_PIXEL * width_px * height_px

    positions = formation_positions_km(satellite, t, time_offsets_seconds)
    target = nadir_target_km(positions[NADIR_REFERENCE_INDEX])

    sza_deg, saa_deg = solar_angles_deg(sun_position_km(eph, t), target)
    if sza_deg >= 90.0:
        raise ValueError(
            f"Sun is below the horizon at the formation's target "
            f"(solar zenith angle={sza_deg:.1f} deg); nothing to check."
        )
    wavelength_nm, atmosphere, surface, grid_3d = build_smartg_scene_3d(
        geocentric_latitude_deg(target)
    )
    atmosphere_profile = atmosphere.calc(wavelength_nm)

    pos_z = grid_3d.zGRID[-1] - SMARTG_3D_SENSOR_TOP_OFFSET_KM
    xgrid = np.linspace(grid_3d.xgrid[0], grid_3d.xgrid[-1], width_px + 1)
    ygrid = np.linspace(grid_3d.ygrid[0], grid_3d.ygrid[-1], height_px + 1)
    cell_size = xgrid[1] - xgrid[0]

    vza_deg, vaa_deg = viewing_angles_deg(positions[satellite_index], target)
    sensors = get_sensors_grid(
        xgrid,
        ygrid,
        pos_z=pos_z,
        th_deg=180.0 - vza_deg,
        ph_deg=(vaa_deg + 180.0) % 360.0,
        loc="ATMOS",
        cell_size=cell_size,
        grid_3d=grid_3d,
    )
    le = LocalEstimate(
        th_deg=np.array([sza_deg]),
        phi_deg=np.array([saa_deg]),
        count_level=np.array([0]),  # 0 = UPTOA, see smartg.sensor.Sensor
    )

    sg = Smartg(back=True, opt3d=True)

    def mean_reflectance(s_min: int, s_max: int) -> float:
        ds = sg.run(
            wavelength=wavelength_nm,
            atmosphere=atmosphere_profile,
            surface=surface,
            sensor=sensors,
            le=le,
            n_photons=n_photons,
            s_min=s_min,
            s_max=s_max,
            seed=123,  # fixed: the three runs must be directly comparable
            progress=False,
        )
        toa_reflectance = ds["I_up (TOA)"].isel(
            **{"Azimuth angles": 0, "Zenith angles": 0}
        )
        return float(toa_reflectance.values.mean())

    max_interactions = SMARTG_SCATTERING_ORDER_MAX_INTERACTIONS
    all_orders_mean = mean_reflectance(s_min=0, s_max=max_interactions)
    single_mean = mean_reflectance(s_min=1, s_max=1)
    multi_mean = mean_reflectance(s_min=2, s_max=max_interactions)

    if all_orders_mean <= 0.0:
        raise ValueError(
            "No signal received in this view (all-orders mean reflectance "
            "is zero); can't compute scattering-order fractions."
        )

    return {
        "all_orders_mean": all_orders_mean,
        "single_scatter_fraction": single_mean / all_orders_mean,
        "multi_scatter_fraction": multi_mean / all_orders_mean,
    }


def reflectance_to_image(
    reflectance: np.ndarray, gain: float = 1.0, gamma: float = 1.0
) -> Image.Image:
    """Convert a (h, w, n_wavelengths >= 3) SMART-G reflectance array
    (dimensionless, roughly 0-1 for a sunlit cloud/atmosphere scene) into a
    displayable 8-bit RGB image: pick out (red, green, blue) via
    SMARTG_RGB_CHANNEL_ORDER, apply a brightness `gain` and a display
    `gamma` correction (`output = (gain * reflectance) ** (1 / gamma)`),
    then clip to [0, 1] and scale to [0, 255]."""
    rgb = reflectance[..., SMARTG_RGB_CHANNEL_ORDER]
    scaled = np.clip(rgb * gain, 0.0, None) ** (1.0 / gamma)
    scaled = np.clip(scaled, 0.0, 1.0)
    return Image.fromarray((scaled * 255).astype(np.uint8), mode="RGB")


def save_formation_images(
    images: dict[int, np.ndarray],
    output_dir: Path,
    gain: float = 1.0,
    gamma: float = 1.0,
) -> list[Path]:
    """Save each satellite's rendered reflectance array (see
    render_formation_images()) as a numbered PNG in `output_dir`. Returns
    the list of written file paths, in satellite index order."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in sorted(images):
        path = output_dir / f"satellite_{index:02d}.png"
        reflectance_to_image(images[index], gain=gain, gamma=gamma).save(path)
        paths.append(path)
    return paths


def add_earth(scene_model, texture_path: Path | None) -> None:
    """Add an Earth sphere to the scene, textured with a world map when
    available, otherwise a plain colored sphere."""
    if texture_path is not None:
        sphere = tvtk.TexturedSphereSource(
            radius=EARTH_RADIUS_KM, theta_resolution=120, phi_resolution=120
        )
        reader = tvtk.JPEGReader(file_name=str(texture_path))
        texture = tvtk.Texture(input_connection=reader.output_port, interpolate=1)
        mapper = tvtk.PolyDataMapper(input_connection=sphere.output_port)
        actor = tvtk.Actor(mapper=mapper, texture=texture)
        # Pole + prime-meridian correction so the (mirrored) texture matches
        # ITRS axes - see ensure_earth_texture() for the derivation.
        actor.rotate_y(180)
        scene_model.add_actor(actor)
        return

    phi, theta = np.mgrid[0 : np.pi : 101j, 0 : 2 * np.pi : 101j]
    x = EARTH_RADIUS_KM * np.sin(phi) * np.cos(theta)
    y = EARTH_RADIUS_KM * np.sin(phi) * np.sin(theta)
    z = EARTH_RADIUS_KM * np.cos(phi)
    scene_model.mlab.mesh(x, y, z, color=(0.2, 0.45, 0.85))


def format_duration(total_seconds: float) -> str:
    total_seconds = max(0, int(total_seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class OrbitSimulation(HasTraits):
    """Mayavi scene, embedded in a TraitsUI window, with buttons to control
    the simulation speed relative to real time and a status widget showing
    how simulated time compares to elapsed real time."""

    scene = Instance(MlabSceneModel, ())

    status_text = Str()

    slower_button = Button("Slower (÷2)")
    pause_button = Button("Pause / Resume")
    faster_button = Button("Faster (×2)")
    reset_button = Button("Reset speed")
    view_button = Button("Toggle Sat 5 POV")
    footprint_selection = Enum(["None"] + [f"Sat {i}" for i in range(NUM_SATELLITES)])

    # SMART-G 3D radiative-transfer image acquisition (see
    # _acquire_images_button_fired()).
    acquire_images_button = Button("Acquire Images (SMART-G)")
    image_resolution = Enum(list(SMARTG_3D_RESOLUTION_PRESETS))
    # All satellites selected by default; unchecking some is the fast lever
    # (see render_formation_images_3d()'s satellite_indices).
    satellites_to_image = List(Str, [f"Sat {i}" for i in range(NUM_SATELLITES)])
    acquisition_status_text = Str()

    view = View(
        VGroup(
            Item(
                "scene",
                editor=SceneEditor(scene_class=MayaviScene),
                height=700,
                width=1000,
                show_label=False,
            ),
            HGroup(
                Item("slower_button", show_label=False),
                Item("pause_button", show_label=False),
                Item("faster_button", show_label=False),
                Item("reset_button", show_label=False),
                Item("view_button", show_label=False),
            ),
            HGroup(
                Item(
                    "footprint_selection",
                    label="Show FOV footprint for",
                    visible_when="following_nadir",
                ),
            ),
            Item("status_text", show_label=False, style="readonly"),
            VGroup(
                HGroup(
                    Item("image_resolution", label="Image resolution"),
                    Item("acquire_images_button", show_label=False),
                ),
                Item(
                    "satellites_to_image",
                    label="Satellites to image",
                    style="custom",
                    editor=CheckListEditor(
                        values=[f"Sat {i}" for i in range(NUM_SATELLITES)],
                        cols=NUM_SATELLITES,
                    ),
                ),
                Item("acquisition_status_text", show_label=False, style="readonly"),
                label="SMART-G image acquisition",
                show_border=True,
            ),
        ),
        resizable=True,
        title="CloudCT Orbit Simulation",
    )

    def __init__(
        self,
        satellite: EarthSatellite,
        t0,
        default_speed: float,
        eph,
        **traits,
    ) -> None:
        super().__init__(**traits)
        self.satellite = satellite
        self.t0 = t0
        self.default_speed = default_speed
        self.eph = eph
        self.time_offsets_seconds = formation_time_offsets_seconds(satellite, t0)

        self.speed = default_speed
        self.paused = False
        self.sim_elapsed_seconds = 0.0
        self.real_elapsed_seconds = 0.0
        self.following_nadir = False
        # Updated every tick from the current simulated UTC time; not
        # visualized yet, just kept available for later use.
        self.sun_position_km = sun_position_km(eph, t0)

        self._satellite_points = None
        self._orbit_line = None
        self._pointing_arrows = None
        self._footprint_surface = None
        self._last_wall_time = None
        self._timer = None
        self._saved_camera = None
        self._refresh_status_text()

    @observe("scene.activated")
    def _setup_scene(self, event=None) -> None:
        self.scene.background = (0.0, 0.0, 0.0)

        texture_path = ensure_earth_texture()
        add_earth(self.scene, texture_path)

        positions = formation_positions_km(
            self.satellite, self.t0, self.time_offsets_seconds
        )
        loop = orbit_plane_loop_km(self.satellite, self.t0)

        self._orbit_line = self.scene.mlab.plot3d(
            loop[:, 0],
            loop[:, 1],
            loop[:, 2],
            color=(1.0, 0.85, 0.0),
            tube_radius=None,
            line_width=1.5,
        )

        self._satellite_points = self.scene.mlab.points3d(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            color=(1.0, 0.1, 0.1),
            scale_factor=SATELLITE_MARKER_SCALE_KM,
        )

        pointing_vectors = formation_pointing_vectors_km(positions)
        self._pointing_arrows = self.scene.mlab.quiver3d(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            pointing_vectors[:, 0],
            pointing_vectors[:, 1],
            pointing_vectors[:, 2],
            color=(0.2, 1.0, 1.0),
            mode="arrow",
            scale_factor=1.0,
            scale_mode="vector",
        )

        self.scene.mlab.title(
            f"{self.satellite.name} formation ({NUM_SATELLITES} satellites)",
            color=(1, 1, 1),
            size=0.35,
        )

        initial_footprint = fov_footprint_km(
            positions[NADIR_REFERENCE_INDEX],
            nadir_target_km(positions[NADIR_REFERENCE_INDEX]),
            orbital_normal_itrs(self.satellite, self.t0),
        )
        self._footprint_surface = self.scene.mlab.mesh(
            initial_footprint[:, :, 0],
            initial_footprint[:, :, 1],
            initial_footprint[:, :, 2],
            color=(1.0, 0.55, 0.0),
            opacity=0.45,
        )
        self._footprint_surface.visible = False

        self._last_wall_time = time.time()
        self._timer = Timer(ANIMATION_DELAY_MS, self._on_tick)

    def _on_tick(self) -> None:
        if self.scene.renderer is None:
            # The window has been closed and its VTK renderer torn down,
            # but this Timer keeps firing on its own schedule regardless -
            # nothing else stops it. Stop it now instead of touching a dead
            # scene (which raises on the camera/mlab_source calls below).
            self._timer.Stop()
            return

        now = time.time()
        dt_real = now - self._last_wall_time
        self._last_wall_time = now

        if not self.paused:
            self.real_elapsed_seconds += dt_real
            self.sim_elapsed_seconds += dt_real * self.speed
            t = self.t0 + self.sim_elapsed_seconds / 86400.0

            positions = formation_positions_km(self.satellite, t, self.time_offsets_seconds)
            # .reset() (not .set()) - in-place .set() point updates don't
            # reliably show up on screen in this app (confirmed for the
            # orbit line; using .reset() everywhere here to stay safe).
            self._satellite_points.mlab_source.reset(
                x=positions[:, 0], y=positions[:, 1], z=positions[:, 2]
            )

            pointing_vectors = formation_pointing_vectors_km(positions)
            self._pointing_arrows.mlab_source.reset(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                u=pointing_vectors[:, 0],
                v=pointing_vectors[:, 1],
                w=pointing_vectors[:, 2],
            )

            loop = orbit_plane_loop_km(self.satellite, t)
            self._orbit_line.mlab_source.reset(x=loop[:, 0], y=loop[:, 1], z=loop[:, 2])

            if self.following_nadir:
                self._apply_nadir_camera(positions[NADIR_REFERENCE_INDEX], t)

            self._update_footprint(positions, t)

            self.sun_position_km = sun_position_km(self.eph, t)

        self._refresh_status_text()

    def _current_time(self):
        return self.t0 + self.sim_elapsed_seconds / 86400.0

    def _apply_nadir_camera(self, sat4_position_km: np.ndarray, t) -> None:
        view_up = orbital_normal_itrs(self.satellite, t)
        cam_pos, focal_point, up = nadir_camera_frame_km(sat4_position_km, view_up)
        camera = self.scene.camera
        camera.position = tuple(cam_pos)
        camera.focal_point = tuple(focal_point)
        camera.view_up = tuple(up)
        self.scene.renderer.reset_camera_clipping_range()
        self.scene.render()

    def _update_footprint(self, positions: np.ndarray, t) -> None:
        """Show the selected satellite's FOV footprint on Earth's surface -
        only meaningful (and only shown) in Sat 5 POV mode."""
        if not self.following_nadir or self.footprint_selection == "None":
            self._footprint_surface.visible = False
            return

        selected_index = int(self.footprint_selection.split()[1])
        view_up = orbital_normal_itrs(self.satellite, t)
        footprint = fov_footprint_km(
            positions[selected_index],
            nadir_target_km(positions[NADIR_REFERENCE_INDEX]),
            view_up,
        )
        self._footprint_surface.mlab_source.reset(
            x=footprint[:, :, 0], y=footprint[:, :, 1], z=footprint[:, :, 2]
        )
        self._footprint_surface.visible = True

    def _refresh_status_text(self) -> None:
        state = "PAUSED" if self.paused else "RUNNING"
        view_mode = "Sat 5 POV" if self.following_nadir else "Free view"
        self.status_text = (
            f"[{state}]  Speed: {self.speed:,.0f}x real-time   |   "
            f"Simulated elapsed: {format_duration(self.sim_elapsed_seconds)}   |   "
            f"Real elapsed: {format_duration(self.real_elapsed_seconds)}   |   "
            f"{view_mode}"
        )

    def _slower_button_fired(self) -> None:
        self.speed = max(MIN_SPEED_MULTIPLIER, self.speed / SPEED_STEP_FACTOR)
        self._refresh_status_text()

    def _faster_button_fired(self) -> None:
        self.speed = min(MAX_SPEED_MULTIPLIER, self.speed * SPEED_STEP_FACTOR)
        self._refresh_status_text()

    def _pause_button_fired(self) -> None:
        self.paused = not self.paused
        self._refresh_status_text()

    def _reset_button_fired(self) -> None:
        self.speed = self.default_speed
        self._refresh_status_text()

    def _view_button_fired(self) -> None:
        camera = self.scene.camera
        self.following_nadir = not self.following_nadir

        # Standing exactly at satellite 4 looking outward, the formation
        # itself shouldn't be visible - you can't see your own satellite
        # (the camera sits right inside its marker) or usefully see the
        # others from this POV, only Earth.
        self._satellite_points.visible = not self.following_nadir
        self._pointing_arrows.visible = not self.following_nadir

        t = self._current_time()
        positions = formation_positions_km(self.satellite, t, self.time_offsets_seconds)

        if self.following_nadir:
            # Entering the satellite's POV: remember exactly where the
            # free-look camera was, so toggling back restores it.
            self._saved_camera = (
                tuple(camera.position),
                tuple(camera.focal_point),
                tuple(camera.view_up),
            )
            self._apply_nadir_camera(positions[NADIR_REFERENCE_INDEX], t)
        elif self._saved_camera is not None:
            position, focal_point, view_up = self._saved_camera
            camera.position = position
            camera.focal_point = focal_point
            camera.view_up = view_up
            self.scene.renderer.reset_camera_clipping_range()
            self.scene.render()

        self._update_footprint(positions, t)
        self._refresh_status_text()

    def _footprint_selection_changed(self) -> None:
        if self._footprint_surface is None:
            return  # fires once at init, before the scene/mesh exist
        t = self._current_time()
        positions = formation_positions_km(self.satellite, t, self.time_offsets_seconds)
        self._update_footprint(positions, t)

    def _acquire_images_button_fired(self) -> None:
        """Freeze the simulation at its current moment and run a SMART-G 3D
        radiative-transfer render (see render_formation_images_3d()) for
        the checked satellites, each from its own point of view of the 3D
        cloud field centred on the formation's shared nadir target - the
        "5th satellite"'s nadir vector to Earth's centre (see
        NADIR_REFERENCE_INDEX), over the field's own fixed domain size
        (see build_smartg_scene_3d()). Runs synchronously: the window
        won't respond to input until it finishes (progress is still shown,
        via GUI.process_events() between satellites), which for the higher
        resolution presets can be a long time - see
        SMARTG_3D_RESOLUTION_PRESETS."""
        if not self.satellites_to_image:
            self.acquisition_status_text = "Select at least one satellite first."
            return

        # "Freezes the simulation time": stop the animation, and render at
        # whatever moment it was showing when the button was pressed.
        self.paused = True
        self._refresh_status_text()

        satellite_indices = sorted(
            int(label.split()[1]) for label in self.satellites_to_image
        )
        width_px, height_px = SMARTG_3D_RESOLUTION_PRESETS[self.image_resolution]
        t = self._current_time()

        def report_progress(done: int, total: int) -> None:
            self.acquisition_status_text = (
                f"Rendering satellite {done}/{total} at {width_px}x{height_px}..."
            )
            GUI.process_events()

        report_progress(0, len(satellite_indices))
        try:
            images = render_formation_images_3d(
                self.satellite,
                t,
                self.time_offsets_seconds,
                self.eph,
                width_px=width_px,
                height_px=height_px,
                satellite_indices=satellite_indices,
                progress_callback=report_progress,
            )
            output_dir = (
                SMARTG_IMAGES_DIR / f"acquisition_{t.utc_strftime('%Y%m%dT%H%M%SZ')}"
            )
            paths = save_formation_images(
                images, output_dir, gain=SMARTG_3D_DEFAULT_GAIN, gamma=1.8
            )
        except Exception as exc:  # a GPU/auxdata/SMART-G failure shouldn't crash the GUI
            self.acquisition_status_text = f"Acquisition failed: {exc}"
        else:
            self.acquisition_status_text = f"Saved {len(paths)} image(s) to {output_dir}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CloudCT formation orbit simulation.")
    parser.add_argument(
        "--download-auxdata",
        action="store_true",
        help=(
            "Download SMART-G's auxiliary datasets (aerosols, clouds, gas "
            "absorption tables, etc.) into smartg_auxdata/ before starting "
            "the simulation. Off by default: can be a large download and "
            "isn't otherwise used by this app. Already-downloaded files are "
            "skipped."
        ),
    )
    parser.add_argument(
        "--auxdata-type",
        nargs="+",
        default="all",
        metavar="TYPE",
        help=(
            "One or more SMART-G auxdata types to download (e.g. aer cld "
            "reptran), only used with --download-auxdata. Defaults to 'all'."
        ),
    )
    parser.add_argument(
        "--render-images",
        action="store_true",
        help=(
            "Render the image each satellite in the formation acquires, at "
            "the simulation's start time, via SMART-G radiative transfer "
            "through the scene's atmosphere and cloud layer, and save them "
            "as PNGs before opening the 3D view. Off by default: requires "
            "a CUDA-capable GPU and the 'atm'/'cld' SMART-G auxdata (see "
            "--download-auxdata) already downloaded, and can take a while."
        ),
    )
    parser.add_argument(
        "--render-mode",
        choices=["uniform", "3d"],
        default="uniform",
        help="Cloud scene for --render-images. 'uniform' (default): a flat, "
        "horizontally-uniform synthetic cloud layer - fast, one shared "
        "simulation for the whole formation. '3d': a real LES-simulated "
        "cumulus cloud field with actual spatial structure - one simulated "
        "view per satellite (still a single Smartg.run() call), needs the "
        "'IPRT' SMART-G auxdata and is slower; see "
        "render_formation_images_3d() for its approximations.",
    )
    parser.add_argument(
        "--render-output-dir",
        type=Path,
        default=SMARTG_IMAGES_DIR,
        metavar="DIR",
        help="Where to save rendered images, only used with --render-images. "
        f"Defaults to {SMARTG_IMAGES_DIR.name}/.",
    )
    parser.add_argument(
        "--render-n-photons",
        type=float,
        default=None,
        metavar="N",
        help="Photon count for the render's SMART-G simulation, only used "
        "with --render-images: higher is less noisy but slower. Defaults "
        f"to {SMARTG_RENDER_N_PHOTONS:.0g} for --render-mode uniform; for "
        f"3d, scales with pixel count instead (SMARTG_3D_PHOTONS_PER_PIXEL "
        f"x width x height) unless given here.",
    )
    parser.add_argument(
        "--render-gain",
        type=float,
        default=None,
        metavar="GAIN",
        help="Brightness multiplier applied to each image's reflectance "
        "before it's saved, only used with --render-images: SMART-G's raw "
        "reflectance output is dim for direct 8-bit display. Defaults to "
        f"3.0 for --render-mode uniform, {SMARTG_3D_DEFAULT_GAIN:g} for 3d "
        "(the non-periodic 3D domain runs dimmer, see build_smartg_scene_3d()).",
    )
    parser.add_argument(
        "--render-gamma",
        type=float,
        default=1.8,
        metavar="GAMMA",
        help="Display gamma correction applied to each image, only used "
        "with --render-images. Defaults to 1.8.",
    )
    parser.add_argument(
        "--check-scattering-order",
        action="store_true",
        help=(
            "Report whether photons in the 3D cumulus scene tend to reach "
            "a satellite's view after scattering once or multiple times "
            "(via Smartg.run()'s s_min/s_max interaction-count filter - "
            "see estimate_scattering_order_fractions()). Off by default; "
            "needs a CUDA-capable GPU and the SMART-G auxdata already "
            "downloaded, including 'IPRT'."
        ),
    )
    parser.add_argument(
        "--check-scattering-order-satellite",
        type=int,
        default=NADIR_REFERENCE_INDEX,
        metavar="INDEX",
        help="Which satellite's view to check (0-"
        f"{NUM_SATELLITES - 1}), only used with --check-scattering-order. "
        f"Defaults to {NADIR_REFERENCE_INDEX} (the formation's shared "
        "nadir-viewing reference satellite).",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    try:
        tle_text = get_tle()
    except requests.RequestException as exc:
        print(f"Failed to request TLE data: {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    print(tle_text)
    print()

    satellite = build_satellite(tle_text)
    print_current_state(satellite)

    ts = load.timescale()
    t0 = ts.now()
    period_seconds = orbital_period_days(satellite) * 86400.0
    default_speed = period_seconds / ORBIT_ANIMATION_SECONDS

    eph = load_ephemeris()

    if args.download_auxdata:
        ensure_smartg_auxdata(data_type=args.auxdata_type)

    time_offsets_seconds = formation_time_offsets_seconds(satellite, t0)

    if args.render_images:
        print(
            f"Rendering formation images via SMART-G "
            f"(--render-mode {args.render_mode}, this can take a while)..."
        )
        try:
            if args.render_mode == "3d":
                # None -> render_formation_images_3d's own per-pixel-scaled
                # default.
                images = render_formation_images_3d(
                    satellite,
                    t0,
                    time_offsets_seconds,
                    eph,
                    n_photons=args.render_n_photons,
                )
                gain = args.render_gain
                if gain is None:
                    gain = SMARTG_3D_DEFAULT_GAIN
            else:
                n_photons = args.render_n_photons
                if n_photons is None:
                    n_photons = SMARTG_RENDER_N_PHOTONS
                images = render_formation_images(
                    satellite, t0, time_offsets_seconds, eph, n_photons=n_photons
                )
                gain = args.render_gain
                if gain is None:
                    gain = 3.0
            paths = save_formation_images(
                images,
                args.render_output_dir,
                gain=gain,
                gamma=args.render_gamma,
            )
        except Exception as exc:  # a GPU/auxdata/SMART-G failure shouldn't block the 3D view
            print(f"Could not render formation images ({exc}).", file=sys.stderr)
        else:
            print(f"Saved {len(paths)} image(s) to {args.render_output_dir}/")

    if args.check_scattering_order:
        satellite_index = args.check_scattering_order_satellite
        print(
            f"Checking scattering order for satellite {satellite_index} "
            "(3 SMART-G runs)..."
        )
        try:
            fractions = estimate_scattering_order_fractions(
                satellite, t0, time_offsets_seconds, eph, satellite_index=satellite_index
            )
        except Exception as exc:  # a GPU/auxdata/SMART-G failure shouldn't block the 3D view
            print(f"Could not check scattering order ({exc}).", file=sys.stderr)
        else:
            print(
                f"Satellite {satellite_index}: single-scatter "
                f"{fractions['single_scatter_fraction']:.1%}, multi-scatter "
                f"{fractions['multi_scatter_fraction']:.1%} of the signal "
                f"(all-orders mean reflectance: {fractions['all_orders_mean']:.5f})"
            )

    app = OrbitSimulation(satellite, t0, default_speed, eph)
    app.configure_traits()


if __name__ == "__main__":
    main()
