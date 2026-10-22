# 4. Reflectance, clouds and surfaces

## Bidirectional reflectance factor

The raw quantity saved by the renders is the TOA reflectance

    ρ(θs, θv, Δφ) = π L / (μs F0),       μs = cos θs

- `L`: upwelling radiance toward the sensor.
- `F0`: solar irradiance at TOA, perpendicular to the beam.
- `θs`: solar zenith (SZA), `θv`: view zenith, `Δφ`: relative azimuth
  between the Sun and the view.

A perfectly diffuse (Lambertian) white surface lit by the Sun gives
`ρ = 1`. A thick cloud gives ρ of roughly 0.5-0.9 depending on geometry; it
can exceed 1 in the glory/forward directions or where cloud sides catch
extra sunlight in 3D. Ocean under a clear sky is dark (ρ ≈ 0.02-0.1).

Because ρ is small for dark scenes and can exceed 1, images are scaled by
`display_gain` before 8-bit conversion (`reflectance_to_image`), and
`reflectance_stats` reports the raw numbers so the gain can be chosen
knowingly. Compare raw data, not the gained PNG.

## Scattering angle and relative azimuth

The scattering angle Θ between the incoming solar ray and the outgoing view
direction:

    cos Θ = -cos θs cos θv + sin θs sin θv cos Δφ

with Δφ = 0 meaning the sensor looks along the Sun's direction of travel
(forward scattering, Θ = 0) and Δφ = 180° with θv = θs meaning exact
backscatter (Θ = 180°). The exact azimuth convention depends on the code
(see `_wrap_toa_azimuth`); check it before comparing to other tools.
Sampling many Θ across a cloud scene separates phase-function features
(cloudbow, glory) from geometric 3D effects.

## Optical thickness and cloud reflectance

For a plane-parallel conservative-scattering cloud, the diffuse reflectance
at normal incidence grows with optical thickness `τ` and with decreasing
asymmetry parameter `g` (two-stream result):

    R ≈ (1 - g) τ / (2 + (1 - g) τ)

Take-aways: reflectance saturates for `τ ≳ 30-50`; thin clouds (`τ < 5`)
let the surface signal through.

## Cloud geometry in 3D

The 3D scene is a gridded field of cumulus-like cloud: a voxel grid of
extinction coefficient plus a droplet effective radius (`build_scene_3d`).
Effects absent in plane-parallel models:

- **Shadows**: cloud shadows on the surface and on neighbouring clouds,
  displaced from the cloud by `h tan θs`.
- **Sidelighting**: cloud sides are bright on the Sun side, dark on the
  shadowed side.
- **Radiative smoothing and channelling**: horizontal photon transport
  blurs small-scale structure.
- **Parallax / layover**: tall clouds appear displaced with view angle by
  `h tan θv` (see file 5).

## Surface models

| Setting | Model | Behaviour |
|---|---|---|
| `surface = ocean` | Cox-Munk glint model, wind speed `wind_speed_ms` | rough sea surface whose slope variance grows with wind (σ² ≈ 0.003 + 0.00512 U, U in m/s); specular glint where the Sun's mirror direction meets the view |
| Lambertian | constant albedo | direction-independent |

**Sun glint**: the glint centre lies where the view direction mirrors the
Sun (`θv = θs`, `Δφ = 180°` in the convention above). Viewing within
~20-30° of that makes the ocean bright and hard to separate from clouds; a
retrieval chooses view angles to avoid or exploit it.

## Atmosphere in the images

Above and below the cloud, the atmosphere adds a veil of molecular
(Rayleigh) and aerosol scattering. Over dark ocean at nadir, this "path
radiance" can be a large fraction of the ocean signal, and it grows at
oblique views because of the longer path. Absorption (ozone, water vapour)
in the chosen band attenuates the signal.

## Further reading

Cox & Munk (1954); Marshak & Davis (eds.), *3D Radiative Transfer in Cloudy
Atmospheres*.
