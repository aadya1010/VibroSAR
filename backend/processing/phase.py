"""Arctangent phase demodulation for raw-phase radar (Mode A).

Why circle fit: the slow-time samples of a moving chest trace an arc/circle in the
IQ plane around a centre = (static clutter + front-end DC). Subtracting the
*mean* is wrong when the arc is partial or the phase swing is large (24 GHz, mm
motion => several radians). We fit the circle centre (Kasa algebraic fit), shift
to it, then unwrap the angle. Assumes small IQ imbalance (ellipse ~ circle).
Failure cases: signal amplitude << noise (fit meaningless), huge motion
(multiple wraps per sample => unwrap errors). Callers must check `fit_quality`.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def fit_circle(z: np.ndarray) -> Tuple[complex, float]:
    """Kasa least-squares circle fit. Returns (centre, radius)."""
    x, y = z.real, z.imag
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    r2 = sol[2] + cx ** 2 + cy ** 2
    return complex(cx, cy), float(np.sqrt(max(r2, 0.0)))


def demodulate(z: np.ndarray, wavelength_m: float) -> Tuple[np.ndarray, dict]:
    """Return displacement in mm (zero-mean) and quality info.

    quality: radius (signal amplitude), fit_residual (rms distance to circle / radius, lower is better).
    """
    c, r = fit_circle(z)
    w = z - c
    resid = float(np.sqrt(np.mean((np.abs(w) - r) ** 2)) / (r + 1e-12))
    ph = np.unwrap(np.angle(w))
    disp_mm = (ph - ph.mean()) * wavelength_m / (4 * np.pi) * 1000.0
    return disp_mm, {"radius": r, "fit_residual": resid}
