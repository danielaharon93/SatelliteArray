# 5. Multi-angle imaging and cloud tomography

## Motivation

A single image of a cloud gives a 2D projection of a 3D object. To estimate
its internal structure (extinction, droplet size, height) the scene must be
observed from **many directions at nearly the same time**. CloudCT-style
missions achieve this with a formation of small satellites that all look at
one target while passing over it. The along-track spacing gives each
satellite a different view angle.

## View angle from the formation geometry

Neighbouring satellites separated along-track by `d` (≈ 100 km), at
altitude `H` (~500-600 km), all aiming at one ground point. For satellite
`k` at along-track displacement `x_k` from the point's nadir, to first order
(flat Earth):

    ψ_k ≈ arctan(x_k / H)

With the aim at the centre of the formation, the outer satellites look from
tens of degrees off-nadir in opposite directions. Include Earth curvature
(file 2) for accurate numbers; the code computes angles exactly from the
orbit.

All views share the same Sun, so their scattering angles Θ differ only
through the viewing direction. The Sun-synchronous orbit fixes the local
solar time, which determines whether the views lie near the principal plane
(containing the Sun and zenith) or cut across it.

## Why many angles help

1. **Parallax / stereo**: a feature at height `h` appears shifted by
   `h tan θv` along the view azimuth, so the disparity between two views
   with zenith angles θ1, θ2 gives height: `h = Δx / (tan θ1 - tan θ2)`.
2. **Phase-function sampling**: different Θ sample the droplet size
   distribution (cloudbow position and width relate to `r_eff` and its
   variance).
3. **Tomography**: with enough views the 3D extinction field can be
   reconstructed by solving an inverse problem: minimise the mismatch
   between measured images and a forward model (3D RT, e.g. SMART-G, or a
   faster approximation) over the voxel extinctions.
4. **Space carving** (initialisation): a voxel is outside the cloud if any
   view ray through it hits a pixel classified as clear. The intersection
   over all views bounds the cloud shape.

The renders produced by this project play the role of the "measured" images
in synthetic studies: simulate known clouds, add noise and pointing error,
run the retrieval, compare to the truth.

## Co-registration

To combine the images, each must be mapped to common scene coordinates,
which depends on where each camera actually pointed. Pointing errors
(file 6) therefore turn directly into registration errors in the
reconstruction. An angular error `δ` moves the image on a surface at slant
range `R` by about `R δ`; 0.05° at 600 km is ≈ 0.5 km, several pixels if
the ground sample distance is ~100 m.

## Time differences

Satellites reach the target at different times, separated by `Δt = d / v`
(≈ 13 s per 100 km). Over the nine gaps of a 10-satellite formation that is
about two minutes, during which clouds evolve and advect (wind of ~10 m/s
moves a cloud ~1 km in 100 s). The renders use a *frozen* cloud field,
which is an idealisation a real retrieval has to relax.

## Further reading

Levis, Schechner, Aides, Davis, *Airborne three-dimensional cloud
tomography* (ICCV 2015); Levis et al., *Multiple-scattering microphysics
tomography* (CVPR 2017); CloudCT mission papers from the Technion group.
