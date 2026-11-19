"""The CloudCT camera's normalised spectral responsivity (configs/cloudct/, see its README).

One curve, eta_hat(lambda), peak 1. Used by process_reflectance_gui.py (to rebuild the camera's channel
from a render's per-wavelength data) and by playground_core.py (the "grid_cloudct" band sets, which
simulate the camera's band as one channel). Plain numpy: no SMART-G.
"""

from pathlib import Path

import numpy as np

CLOUDCT_RESPONSE_FILE = Path(__file__).with_name("configs") / "cloudct" / "eta_hat_spectral_responsivity.csv"

# The camera's "support" (the wavelengths a render simulates): where eta_hat is at least this share of
# its peak. Strictly the curve is nonzero from 301 to 1093 nm, but outside about 588-704 nm it is under 1 %
# of the peak (and under 0.1 % outside 400-810 nm), which a render would spend most of its wavelengths on.
SUPPORT_THRESHOLD = 0.01


def load_cloudct_response() -> tuple[np.ndarray, np.ndarray]:
    """The CloudCT camera's (wavelength nm, eta_hat), sorted by wavelength."""
    table = np.genfromtxt(CLOUDCT_RESPONSE_FILE, delimiter=",", names=True)
    nm, eta = table["wavelength_nm"], table["eta_hat"]
    order = np.argsort(nm)
    return nm[order], eta[order]


def support_nm(threshold: float = SUPPORT_THRESHOLD) -> tuple[float, float]:
    """(low, high) in whole nm (rounded outwards): the wavelengths where eta_hat >= `threshold`."""
    nm, eta = load_cloudct_response()
    above = nm[eta >= threshold]
    if above.size == 0:
        raise ValueError(f"the CloudCT response never reaches {threshold:g}")
    return float(np.floor(above.min())), float(np.ceil(above.max()))
