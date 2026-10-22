# 3. Radiative transfer and Monte Carlo (SMART-G)

## Quantities

- **Radiance** `L` [W m⁻² sr⁻¹ nm⁻¹]: power per unit area perpendicular to
  the direction, per unit solid angle. This is what a camera pixel measures
  (up to a calibration constant).
- **Irradiance** `E`: power per unit area from all directions.
- **Solar irradiance** at top of atmosphere (TOA), `F0` (normal incidence).

## The radiative transfer equation (RTE)

Along a path coordinate `s` in direction `Ω`:

    dL(s, Ω)/ds = -k_ext L  +  k_abs B  +  k_sca ∫ p(Ω', Ω) L(Ω') dΩ'/4π

- `k_abs`, `k_sca` [km⁻¹]: absorption and scattering coefficients;
  extinction `k_ext = k_abs + k_sca`.
- `ω0 = k_sca / k_ext`: **single-scattering albedo** (≈ 1 for clouds in the
  visible; they scatter, barely absorb).
- `p(Ω', Ω)`: **phase function**, the angular distribution of scattered
  light, normalised so `∫ p dΩ/4π = 1`.
- **Optical depth** `τ = ∫ k_ext ds`. Direct-beam transmission is
  `exp(-τ)` (Beer-Lambert).

In the visible, thermal emission `B` is negligible; the problem is purely
scattering of sunlight.

## Scattering

| Process | Particles | Phase function | Notes |
|---|---|---|---|
| **Rayleigh** | air molecules (≪ λ) | `p = ¾ (1 + cos²Θ)` (plus depolarisation) | τ ∝ λ⁻⁴; makes the sky blue; weak at red wavelengths |
| **Mie** | cloud droplets, aerosols (~λ or larger) | from Mie theory; strongly forward-peaked | asymmetry `g = ⟨cos Θ⟩ ≈ 0.85` for water clouds |
| **Surface** | ocean / land | BRDF (see file 4) | |

Features of the cloud phase function that the images depend on: the
**glory** (backscatter ring near Θ = 180°), the **cloudbow** (~Θ = 140°)
and the strong forward peak. These make cloud brightness depend strongly on
the viewing/solar geometry, which is the reason for multi-angle imaging.

A cloud's bulk properties are set by the liquid water content (LWC) and the
**effective radius** `r_eff` of the droplets:

    k_ext ≈ 3 LWC / (2 ρ_w r_eff)

Typical cumulus: `r_eff` ≈ 5-15 µm, `k_ext` of order 10-100 km⁻¹.

## Why Monte Carlo

For 3D inhomogeneous clouds the RTE has no closed-form solution, and the
plane-parallel assumption (the cloud is infinite and uniform horizontally)
fails: cloud sides, shadows and gaps change the radiance, and the images
carry exactly that 3D information. **Monte Carlo** treats light as many
photon packets and samples the RTE statistically.

One photon history:

1. Launch at TOA in the solar direction (forward mode) with weight 1.
2. Sample the free path: `s = -ln(ξ) / k_ext`, ξ uniform in (0,1]. In a
   heterogeneous medium use **delta tracking** (null collisions) against a
   majorant `k_max`, so the path need not be integrated exactly.
3. At a collision, scatter with probability `ω0` (or reduce the weight by
   `ω0`); pick the scattering species with probability proportional to its
   `k_sca`; sample the new direction from its phase function.
4. At a surface hit, sample the BRDF.
5. Repeat until the photon exits the domain or is cut off.

### Backward tracing and local estimates

Very few forward photons happen to reach a small sensor. SMART-G's
`back=True` mode traces from the sensor toward the Sun, and uses **local
estimates**: at each collision the probability of scattering directly toward
the Sun (or sensor), attenuated by transmission along that path, is added to
the pixel. This converges far faster than waiting for photons to land in a
pixel.

### Statistical error

The result is an average of `N` weighted samples, so the standard error
falls as `1/√N`. In the image: speckle at low photons-per-pixel, smooth at
high. Halving the noise costs 4× the photons; this is the trade-off behind
the `photons_per_pixel` setting. Noise is highest where the signal comes
from rare multiple-scattering paths (shadowed regions, cloud sides).

## SMART-G specifics

[SMART-G](https://github.com/hygeos/smartg) (Speed-up Monte-Carlo Advanced
Radiative Transfer code using GPU) runs the photon loop on an NVIDIA GPU,
which is why the project needs CUDA. Relevant features:

- Plane-parallel or spherical-shell atmosphere, plus an `opt3d` mode with a
  3D voxel grid for cloud extinction (`build_scene_3d`).
- Periodic horizontal boundaries (the `periodic` setting): photons leaving
  one side re-enter the other, representing an extended cloud field rather
  than an isolated block.
- `s_min` / `s_max` filter photons by interaction count. This is how
  `estimate_scattering_order_fractions` measures single vs multiple
  scattering contributions.
- Output is TOA **reflectance** (see file 4) at the requested sensors.

## Further reading

Mayer (2009), *Radiative transfer in the cloudy atmosphere*; Marchuk et
al., *The Monte Carlo Methods in Atmospheric Optics*; Bohren & Huffman,
*Absorption and Scattering of Light by Small Particles*; Thomas & Stamnes,
*Radiative Transfer in the Atmosphere and Ocean*.
