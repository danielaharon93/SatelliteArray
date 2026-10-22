# 2. Coordinate frames and viewing geometry

## Frames used

| Frame | Origin / axes | Used for |
|---|---|---|
| **TEME** | Earth centre; true equator, mean equinox of date | SGP4 output |
| **ITRS / ECEF** | Earth centre; rotates with the Earth. z = rotation axis, x through (0°N, 0°E) | satellite and target positions, the textured Earth |
| **Geodetic** | latitude φ, longitude λ, height h on the WGS-84 ellipsoid | human-readable locations |
| **ENU** (local) | origin at a ground target; East, North, Up | zenith/azimuth of the Sun and of the satellites |
| **Camera** | origin at the satellite; z along boresight | field of view, pointing error |
| **SMART-G scene** | Cartesian box (x, y, z) in km, z up, around the cloud field | the radiative-transfer rendering |

Converting between them is a rotation (plus a translation for origin
changes). `skyfield.framelib.itrs` handles TEME → ITRS, which includes
Earth rotation, precession/nutation and polar motion.

## Geodetic vs geocentric latitude

On the ellipsoid the vertical does not pass through the Earth centre, so
geodetic latitude φ and geocentric latitude φ' differ by up to ~0.19°:

    tan φ' = (1 - e²) tan φ,      e² ≈ 0.00669

`main.py` models the Earth as a sphere of radius `EARTH_RADIUS_KM`
(6371 km) in the 3D view. That is fine for visualisation; the maximum
radius error (~21 km) is small next to the 100 km satellite spacing. Be
aware of it if you compare angles to ellipsoid-exact tools
(`geocentric_latitude_deg` exists for that reason).

## Local ENU basis and angles

Given a point with position `p` (ECEF) on a sphere:

    Up    u = p / ‖p‖
    East  e = (z × u) / ‖z × u‖        (z = (0,0,1))
    North n = u × e

For a direction `d` (unit vector, from the observer toward the object):

    zenith angle   θ = arccos(d · u)
    azimuth        φ = atan2(d · e, d · n)       (clockwise from north)

This is what `zenith_azimuth_deg` computes, and `viewing_angles_deg` /
`solar_angles_deg` build on it. The same basis defines the scene's x
(east), y (north), z (up) axes in SMART-G.

## Pinhole camera and field of view

Each sensor is a rectilinear pinhole camera with half-angles `αx`, `αy`
(`FOV_HALF_ANGLE_X_DEG`, `FOV_HALF_ANGLE_Y_DEG`). A pixel at normalised
image coordinates `(u, v) ∈ [-1, 1]²` has the ray direction

    d_cam = normalise( u·tan αx,  v·tan αy,  1 )

Rotating `d_cam` into ECEF with the camera frame and intersecting with the
Earth gives the pixel's ground point. The ground sample distance grows
toward the image edges and with off-nadir angle: for a satellite at
altitude `H` and off-nadir angle `ψ`, the slant range is roughly
`H / cos ψ` and the pixel footprint is stretched by about `1/cos ψ` across
the line of sight and `1/cos² ψ` along it (flat-Earth approximation).

## Ray-sphere intersection

A ray `p(s) = o + s·d` (`‖d‖ = 1`) hits a sphere of radius `R` at the
roots of

    s² + 2 (o·d) s + (‖o‖² - R²) = 0

    s = -(o·d) ± sqrt( (o·d)² - (‖o‖² - R²) )

The nearer positive root is the visible surface; a negative discriminant
means the ray misses the Earth. `ray_sphere_intersection_km` uses this to
draw the footprint patch.

## Off-nadir angle vs view zenith angle

At the surface, the local zenith angle `θ` of the line of sight is larger
than the satellite's off-nadir angle `ψ` because of Earth curvature. By the
sine rule in the Earth-centre / satellite / target triangle:

    sin θ = (r_sat / R) sin ψ

For r_sat/R ≈ 1.09 this is a 9% enlargement (e.g. ψ = 30° → θ ≈ 33°).
*Off-nadir* is measured at the satellite, *view zenith* at the target;
mixing them up is a common source of error. The radiative-transfer
calculation needs the view zenith at the target.

## Aiming point

All satellites aim at one point: the ground target raised to the cloud
field's centre height (`aim_point_km`). Aiming at the mid-height rather
than the surface keeps the cloud field centred in every view even when
looking obliquely, because an oblique ray reaches the cloud at a horizontal
offset (parallax) from where it reaches the ground. `slant_offset_km`
deals with that offset: for a height `h` and view zenith `θ` it is
`h tan θ`.
