# 6. Pointing-error model

Implemented in [`pointing_noise.py`](../pointing_noise.py). The model
follows the "simple, standard-compliant" recommendation in
`Target_Noise_Model_Document.pdf`, itself based on the ECSS-E-ST-60-10C and
ESSB-HB-E-003 pointing-error handbooks.

## Error budget vocabulary

- **Pointing error**: the angle between the commanded and the true line of
  sight.
- **Performance error** (what is modelled): the true attitude error.
- **Knowledge error**: the error in what the satellite *believes* its
  attitude is. It only matters for geolocating images afterwards, so it is
  not modelled.
- ECSS handbooks classify each term by its *temporal* behaviour (bias,
  random per observation, drift, periodic, random jitter) and combine them
  in a prescribed way.

## The model

Per satellite `s`, acquisition `k` and camera axis `i ∈ {x, y, z}`:

    e_i = b_i^(s) + p_i^(s,k) + g_i(t) + j_i(t)

| Term | Distribution | Time behaviour |
|---|---|---|
| `b` bias | N(0, σb²) | constant for a satellite in one Monte-Carlo run: residual misalignment, calibration error |
| `p` per-acquisition offset | N(0, σp²) | redrawn at every acquisition: controller residual, attitude-dependent star-tracker bias |
| `g` slow drift | first-order Gauss-Markov, std σg, time constant τ | correlated over ~τ: thermal drift, slowly varying sensor errors |
| `j` jitter | N(0, σj²) | uncorrelated between exposures: high-frequency vibration |

Axes `x`, `y` are across the boresight (roll, pitch); `z` is about it
(yaw). Yaw errors are normally 2-5× larger because star trackers measure
rotation about their own boresight poorly; the model multiplies the sigmas
by `yaw_factor`.

### Gauss-Markov drift

With sample step Δt and `φ = exp(-Δt/τ)`:

    g_{n+1} = φ g_n + σg sqrt(1 - φ²) w_n,      w_n ~ N(0, 1)

Started from `g_0 ~ N(0, σg²)`, the variance stays at σg² (a *stationary*
process) and the autocorrelation is `σg² exp(-|Δt|/τ)`. When Δt ≫ τ,
successive samples are independent; when Δt ≪ τ the drift is almost
constant. With the defaults (60 s interval, τ = 600 s) the drift is strongly
correlated between consecutive acquisitions.

### Combining

Independent terms add in variance:

    σ_total² = σb² + σp² + σg² + σj²

For the defaults in `configs/scenario.json` (0.03°, 0.05°, 0.07°, 0.005°)
that is ≈ 0.09° (1σ) per cross-boresight axis, ≈ 0.27° at 3σ, the level at
which ECSS requirements are normally stated.

## Applying the error

The error angles are small, so `(e_x, e_y, e_z)` is treated as a rotation
vector. With `α = ‖e‖` and `k = e / α`, Rodrigues' formula rotates the
commanded line of sight `d`:

    d' = d cos α + (k × d) sin α + k (k·d)(1 - cos α)

The perturbed `d'` is then ray-traced to the scene as in file 2. For small
angles `d' ≈ d + e × d`, a displacement perpendicular to `d`, giving a
ground shift of about `R · ‖e_perp‖` at slant range `R`.

## Monte Carlo use

Each Monte-Carlo run draws a new set of biases and then new `p`, `g`, `j`
per acquisition. Repeating many runs gives the distribution of image
misregistration and, downstream, of retrieval error. `noise_seed` makes a
run reproducible, which matters when comparing retrieval algorithms on
identical inputs.

## What is not modelled

- Sinusoidal reaction-wheel jitter and intra-exposure motion blur: each
  SMART-G render is one static exposure, so jitter is a single random
  sample.
- Satellite position (ephemeris) error, small next to attitude error at
  these scales.
- Time-tag error.

## Further reading

ECSS-E-ST-60-10C *Control performance*; ESSB-HB-E-003 *Pointing error
engineering handbook* (ESA); Wertz (ed.), *Spacecraft Attitude
Determination and Control*.
