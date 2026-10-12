"""Human-body signal models (synthetic): respiration, gross motion, trajectories."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.signal import butter, sosfiltfilt

from simulator.scenario import PersonSpec

X_BOUNDS = (-3.0, 3.0)
Y_BOUNDS = (1.2, 5.5)


def lowpass_noise(rng: np.random.Generator, n: int, fs: float, fc: float) -> np.ndarray:
    """Unit-variance smooth random sequence (zero-phase low-pass of white noise)."""
    sos = butter(2, fc, fs=fs, output="sos")
    x = sosfiltfilt(sos, rng.standard_normal(n + 200))[100:-100]
    return x / (x.std() + 1e-12)


def respiration_mm(t: np.ndarray, f0: float, amp_mm: float, rng: np.random.Generator) -> np.ndarray:
    """Chest displacement: slowly varying rate, 2nd harmonic, amplitude modulation, baseline drift."""
    n, fs = len(t), 1.0 / float(t[1] - t[0])
    f = f0 * (1.0 + 0.06 * lowpass_noise(rng, n, fs, 0.05))
    phase = 2 * np.pi * np.cumsum(f) / fs + rng.uniform(0, 2 * np.pi)
    am = 1.0 + 0.15 * lowpass_noise(rng, n, fs, 0.05)
    x = am * amp_mm * (np.sin(phase) + 0.15 * np.sin(2 * phase + rng.uniform(0, 2 * np.pi)))
    return x + 0.4 * amp_mm * lowpass_noise(rng, n, fs, 0.02)


def motion_bursts_mm(t: np.ndarray, level: float, rng: np.random.Generator
                     ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """Gross body motion (posture shifts, arm movement). Returns (displacement_mm, active_mask, burst_starts_s)."""
    disp, mask, starts = np.zeros_like(t), np.zeros(len(t), bool), []
    T = float(t[-1] - t[0])
    for _ in range(rng.poisson((0.05 + 0.4 * level) * T)):
        t0, dur = rng.uniform(0, T), rng.uniform(1.0, 3.0)
        sel = (t >= t0) & (t < t0 + dur)
        if sel.sum() < 4:
            continue
        tau = (t[sel] - t0) / dur
        comps = sum(np.sin(2 * np.pi * rng.uniform(0.2, 1.5) * (t[sel] - t0) + rng.uniform(0, 6.28))
                    for _ in range(3)) / 3.0
        disp[sel] += level * rng.uniform(5, 40) * np.hanning(len(tau)) * comps
        mask |= sel
        starts.append(t0)
    return disp, mask, starts


def _reflect(x0: float, v: float, t: np.ndarray, lo: float, hi: float) -> np.ndarray:
    L = hi - lo
    m = np.mod(x0 - lo + v * t, 2 * L)
    return lo + np.where(m <= L, m, 2 * L - m)


def trajectory(p: PersonSpec, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Position (x, y) and velocity (vx, vy) over time; people bounce inside the scene box."""
    x = _reflect(p.x_m, p.vx_m_s, t, *X_BOUNDS)
    y = _reflect(p.y_m, p.vy_m_s, t, *Y_BOUNDS)
    dt = float(t[1] - t[0])
    return x, y, np.gradient(x, dt), np.gradient(y, dt)
