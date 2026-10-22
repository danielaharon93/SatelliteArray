# 1. Orbital mechanics, TLEs and SGP4

## Why it matters here

The simulation places a train of satellites on a Sun-synchronous low Earth
orbit (LEO) and needs their positions and the Sun's position at a given UTC
time. Positions come from a published TLE propagated with SGP4 (via
`skyfield`).

## Two-body motion

A satellite of negligible mass around a point-mass Earth obeys

    r'' = -mu r / ‖r‖³,        mu = G M_earth ≈ 398 600.4418 km³/s²

The orbit is a conic; for a bound orbit an ellipse with semi-major axis `a`:

- Period: `T = 2π sqrt(a³ / mu)`
- Vis-viva speed: `v² = mu (2/r - 1/a)`
- Circular orbit: `v = sqrt(mu / r)`

For a ~500-600 km altitude, `a ≈ 6900-7000 km`, giving `T ≈ 95-97 min`
(about 15 revolutions per day, matching the *mean motion* ≈ 14.9 rev/day in
the TLE in `configs/scenario.json`). Orbital speed is ≈ 7.6 km/s.

### Classical orbital elements

| Element | Symbol | Meaning |
|---|---|---|
| Semi-major axis | a | size |
| Eccentricity | e | shape (≈ 0 here) |
| Inclination | i | tilt of the plane against the equator |
| RAAN | Ω | where the plane crosses the equator going north |
| Argument of perigee | ω | orientation of the ellipse in its plane |
| Mean anomaly | M | position along the orbit at epoch |

## Perturbations that matter in LEO

- **J2 (Earth oblateness)** makes the orbital plane precess:

      dΩ/dt = -(3/2) J2 (R_e / p)² n cos i

  Choosing `i ≈ 97-98°` (retrograde) at ~500-600 km makes `dΩ/dt` equal the
  Earth's mean motion around the Sun, ≈ +0.9856°/day. This is a
  **Sun-synchronous orbit**: the satellite crosses a given latitude at the
  same local solar time every pass, so the solar zenith angle at the imaged
  target is nearly repeatable. The TLE inclination 97.7° in
  `configs/scenario.json` is in exactly this regime.
- **Atmospheric drag** slowly lowers the orbit (the `B*` term in the TLE).
  Negligible over the minutes simulated here, but the reason TLEs go stale
  after days to weeks.

## Two-Line Elements and SGP4

A TLE is a compact, fixed-format snapshot of *mean* elements at an epoch,
fitted to the SGP4 propagator. Key fields:

- Line 1: epoch (year + fractional day), first derivative of mean motion,
  `B*` drag term.
- Line 2: inclination, RAAN, eccentricity, argument of perigee, mean
  anomaly, mean motion (rev/day), revolution number.

**SGP4 must be used with TLEs.** The elements are *mean* elements specific
to that model's averaging, so feeding them to a plain Kepler solver gives
errors of several km. Output is in the **TEME** frame, which `skyfield`
converts to other frames. Typical accuracy is ~1 km at epoch, growing by
roughly 1-3 km/day. This is adequate for visualising a formation and
choosing view angles, not for precision pointing.

## Formation along the track

`main.py` puts `NUM_SATELLITES` satellites on the same orbit, separated
along-track by a constant spacing (`SATELLITE_SPACING_KM`). Two equivalent
descriptions:

- Distance `d` along track ⇔ time offset `Δt = d / v`. The satellites
  occupy the same path, a few seconds apart (see
  `formation_time_offsets_seconds`). 100 km at 7.6 km/s is ≈ 13 s.
- Central angle `Δθ = d / r`; e.g. 100 km at r ≈ 6900 km is ≈ 0.83°.

A "frozen" formation holds these offsets fixed while the orbit evolves,
instead of letting separate propagations drift. A real formation needs
station-keeping manoeuvres because differential drag and J2 change the
spacing.

## Sun position

The Sun's direction comes from a planetary ephemeris (JPL DE series loaded
by `skyfield`). Because the Earth-Sun distance (~1.5×10⁸ km) dwarfs the
orbit, the Sun can be treated as a **collimated source** at the target: the
solar zenith angle (SZA) and azimuth are computed once for the imaged
point (`solar_angles_deg`) and applied to the whole scene.

## Further reading

Vallado, *Fundamentals of Astrodynamics and Applications*; Vallado et al.,
*Revisiting Spacetrack Report #3*; Curtis, *Orbital Mechanics for
Engineering Students*.
