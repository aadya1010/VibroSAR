"""Filter helpers. All designs are second-order-section Butterworth (numerically stable)."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt


def bandpass_sos(lo: float, hi: float, fs: float, order: int = 4) -> np.ndarray:
    if not 0 < lo < hi < fs / 2:
        raise ValueError(f"bad band {lo}-{hi} Hz for fs={fs}")
    return butter(order, [lo, hi], btype="band", fs=fs, output="sos")


def highpass_sos(fc: float, fs: float, order: int = 2) -> np.ndarray:
    return butter(order, fc, btype="high", fs=fs, output="sos")


def zero_phase(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """Zero-phase (offline/windowed) filtering. NOT causal: use sosfilt for strict real-time."""
    return sosfiltfilt(sos, x, axis=0)


def causal(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    return sosfilt(sos, x, axis=0)
