"""Camera pointing-error (performance error) model for the satellite
imaging simulation, following the recommended model in
Target_Noise_Model_Document.pdf ("Recommendations: A simple,
standard-compliant noise model", after ECSS-E-ST-60-10C / ESSB-HB-E-003).

Per satellite s, acquisition k and camera-frame axis i in {x, y, z}:

    e_i = b_i^(s) + p_i^(s,k) + g_i(t) + j_i(t)

- b: static bias ~ N(0, sigma_b^2), drawn once per satellite per
  Monte-Carlo run (residual misalignment / calibration error).
- p: per-acquisition offset ~ N(0, sigma_p^2), redrawn at every new
  acquisition (controller residual, attitude-dependent star-tracker bias).
- g: first-order Gauss-Markov drift, g_{n+1} = phi*g_n + sigma_g*sqrt(1-phi^2)*w_n,
  phi = exp(-dt/tau), initialised g_0 ~ N(0, sigma_g^2) so it's stationary.
- j: jitter, band-limited white Gaussian ~ N(0, sigma_j^2) per acquisition.

x and y are across the boresight (roll / pitch), z is about it (yaw). The
sigmas are given per cross-boresight axis; yaw's are `yaw_factor` times
larger (the document: "about-boresight about 2-5x larger").

The error is applied as a small rotation of the commanded line of sight
(exact Rodrigues rotation - the document's step 5 allows it in place of
the error quaternion), which is then ray-traced to the ground.

Not modelled here: knowledge error (what the satellite *believes* it points
at) - it only matters for geolocating the image afterwards, which this
simulation doesn't do; and sinusoidal reaction-wheel jitter / intra-
exposure blur - each SMART-G render is a single static exposure, so the
jitter only enters as one random sample of the line of sight.
"""

from dataclasses import dataclass

import numpy as np

# The document's "Suggested starting parameters" table (1 sigma per
# cross-boresight axis). Engineering assumptions, not published values.
PRESETS = {
    "6U CubeSat (C3IEL-like, ~0.1 deg)": {
        "sigma_bias_deg": 0.03,
        "sigma_acquisition_deg": 0.05,
        "sigma_drift_deg": 0.07,
        "drift_tau_s": 600.0,
        "sigma_jitter_deg": 0.005,
    },
    "Agile EO satellite (~0.01 deg)": {
        "sigma_bias_deg": 0.003,
        "sigma_acquisition_deg": 0.005,
        "sigma_drift_deg": 0.007,
        "drift_tau_s": 1000.0,
        "sigma_jitter_deg": 0.0005,
    },
}
DEFAULT_PRESET = "6U CubeSat (C3IEL-like, ~0.1 deg)"
DEFAULT_YAW_FACTOR = 3.0
DEFAULT_ACQUISITION_INTERVAL_S = 60.0

# Rayleigh statistics for the cone (line-of-sight) error of two equal-sigma
# Gaussian axes: CE90 ~ 2.15 sigma (the document's budget-check step).
CE90_PER_SIGMA = 2.146


@dataclass
class PointingNoiseParams:
    sigma_bias_deg: float
    sigma_acquisition_deg: float
    sigma_drift_deg: float
    drift_tau_s: float
    sigma_jitter_deg: float
    yaw_factor: float = DEFAULT_YAW_FACTOR
    acquisition_interval_s: float = DEFAULT_ACQUISITION_INTERVAL_S

    def ape_sigma_deg(self) -> float:
        """APE 1-sigma per cross-boresight axis: RSS of the four
        (uncorrelated) contributors."""
        return float(
            np.sqrt(
                self.sigma_bias_deg**2
                + self.sigma_acquisition_deg**2
                + self.sigma_drift_deg**2
                + self.sigma_jitter_deg**2
            )
        )


class PointingNoise:
    """Holds one Monte-Carlo run's random state: each satellite's static
    bias and drift state persist across acquisitions, until `reset()`.

    The bias and drift are stored as unit-variance draws and scaled by the
    current sigmas at draw time, so changing a sigma rescales an existing
    run instead of needing a new one."""

    def __init__(self, seed: int = 0) -> None:
        self.reset(seed)

    def reset(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)
        self._acquisition = 0
        self._bias_unit: dict = {}
        self._drift_unit: dict = {}
        self._drift_acquisition: dict = {}

    @property
    def acquisition(self) -> int:
        return self._acquisition

    def begin_acquisition(self) -> None:
        """Start acquisition k+1 (one Render click). Satellites rendered
        together in the same acquisition share its time instant."""
        self._acquisition += 1

    def draw_error_rad(self, key, params: PointingNoiseParams) -> np.ndarray:
        """The pointing error rotation vector (e_x, e_y, e_z) in radians,
        camera frame, for satellite `key` at the current acquisition."""
        rng = self._rng
        if key not in self._bias_unit:
            self._bias_unit[key] = rng.standard_normal(3)
            self._drift_unit[key] = rng.standard_normal(3)  # stationary g_0
        else:
            steps = self._acquisition - self._drift_acquisition[key]
            elapsed_s = steps * params.acquisition_interval_s
            phi = np.exp(-elapsed_s / params.drift_tau_s) if params.drift_tau_s > 0 else 0.0
            self._drift_unit[key] = phi * self._drift_unit[key] + np.sqrt(
                1.0 - phi**2
            ) * rng.standard_normal(3)
        self._drift_acquisition[key] = self._acquisition

        acquisition_unit = rng.standard_normal(3)
        jitter_unit = rng.standard_normal(3)
        error_deg = (
            params.sigma_bias_deg * self._bias_unit[key]
            + params.sigma_acquisition_deg * acquisition_unit
            + params.sigma_drift_deg * self._drift_unit[key]
            + params.sigma_jitter_deg * jitter_unit
        )
        error_deg *= np.array([1.0, 1.0, params.yaw_factor])
        return np.radians(error_deg)


def camera_frame(boresight: np.ndarray, along_track: np.ndarray):
    """Camera (x, y, z) unit axes: z along the boresight, x the along-track
    direction made perpendicular to it (roll axis), y = z cross x."""
    z = boresight / np.linalg.norm(boresight)
    x = along_track - np.dot(along_track, z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return x, y, z


def rotate(vector: np.ndarray, rotation_vector: np.ndarray) -> np.ndarray:
    """Exact (Rodrigues) rotation of `vector` by `rotation_vector` (axis *
    angle in radians)."""
    angle = np.linalg.norm(rotation_vector)
    if angle == 0.0:
        return vector.copy()
    k = rotation_vector / angle
    return (
        vector * np.cos(angle)
        + np.cross(k, vector) * np.sin(angle)
        + k * np.dot(k, vector) * (1.0 - np.cos(angle))
    )


def perturbed_boresight(
    boresight: np.ndarray, along_track: np.ndarray, error_rad: np.ndarray
) -> np.ndarray:
    """The actual line of sight u_true = R_err * u_cmd, for a camera-frame
    error rotation vector `error_rad` (see PointingNoise.draw_error_rad())."""
    x, y, z = camera_frame(boresight, along_track)
    rotation_vector = error_rad[0] * x + error_rad[1] * y + error_rad[2] * z
    u_true = rotate(z, rotation_vector)
    return u_true / np.linalg.norm(u_true)
