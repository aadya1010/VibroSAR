"""NLMS adaptive noise canceller with safety gating (batch version).

Model: target d[n] = survivor/useful + machinery(filtered by unknown path) + noise.
Reference x[n] (sensor near machinery) sees mostly machinery. FIR w (L taps) learns
the machinery path; output e = d - w.x keeps whatever is NOT predictable from x.

Safety design (so we do not eat the useful signal):
  * leakage (w decays) and small step size: slow, conservative adaptation;
  * adapt ONLY when the reference is clearly active (power > gate_k x sensor-noise-floor^2);
  * FREEZE when the target shows a burst the reference does not (a knock at the
    target must not be 'learned' as machinery);
  * pilot-injection test (see pilot_recovery) measures how much of a known weak
    signal survives. If recovery is low, callers must treat the output as unsafe.
Latency: batch/offline here; streaming variant is the same loop with persistent state.
Assumes: reference and target are time-aligned (same ADC clock) and fs identical.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


@dataclass(frozen=True)
class NLMSConfig:
    taps: int = 32
    mu: float = 0.05
    leak: float = 1e-4
    eps: float = 1e-8
    gate_k: float = 4.0         # adapt only if ref power > gate_k * (ref_floor_g)^2
    ref_floor_g: float = 0.005  # reference-sensor noise floor (RMS, g); ABSOLUTE so an always-on machine still adapts
    burst_k: float = 12.0       # target burst if power > burst_k * floor
    ema_s: float = 0.1


def _ema_power(x: np.ndarray, fs: float, tau_s: float) -> np.ndarray:
    a = 1.0 - np.exp(-1.0 / (tau_s * fs))
    p = np.empty_like(x)
    acc = x[0] ** 2
    for i, v in enumerate(x):
        acc += a * (v * v - acc)
        p[i] = acc
    return p


def _floor(p: np.ndarray) -> np.ndarray:
    """Minimum-statistics floor: drops instantly, rises slowly."""
    out = np.empty_like(p)
    f = p[0]
    for i, v in enumerate(p):
        f = min(v, f * 1.0003)
        out[i] = f
    return out


def nlms_cancel(d: np.ndarray, x: np.ndarray, fs: float, cfg: NLMSConfig = NLMSConfig(),
                w0: np.ndarray | None = None) -> Tuple[np.ndarray, dict]:
    """Return (cleaned target, info). d and x are 1-D, already high-passed (no DC)."""
    if len(d) != len(x):
        raise ValueError("target and reference must have equal length")
    L = cfg.taps
    xp = np.concatenate([np.zeros(L - 1), x])
    U = sliding_window_view(xp, L)[:, ::-1]          # row n = [x[n], x[n-1], ..., x[n-L+1]]
    px, pd = _ema_power(x, fs, cfg.ema_s), _ema_power(d, fs, cfg.ema_s)
    fx, fd = _floor(px) + 1e-12, _floor(pd) + 1e-12
    adapt_ok = px > cfg.gate_k * cfg.ref_floor_g ** 2   # NOT relative to a tracked floor: constant machinery would never pass
    freeze = (pd > cfg.burst_k * fd) & ~adapt_ok      # burst at target not explained by reference
    w = np.zeros(L) if w0 is None else w0.copy()
    e = np.empty_like(d)
    n_adapt = 0
    for n in range(len(d)):
        u = U[n]
        y = w @ u
        e[n] = d[n] - y
        if adapt_ok[n] and not freeze[n]:
            w = (1.0 - cfg.leak) * w + cfg.mu * e[n] * u / (u @ u + cfg.eps)
            n_adapt += 1
    return e, {"adapt_frac": n_adapt / len(d), "w": w}


def _lockin(x: np.ndarray, f: float, fs: float) -> complex:
    t = np.arange(len(x)) / fs
    return 2.0 * np.mean(x * np.exp(-2j * np.pi * f * t))


def pilot_recovery(d: np.ndarray, x: np.ndarray, fs: float, cfg: NLMSConfig = NLMSConfig(),
                   f_pilot: float = 47.3, amp_g: float = 0.006) -> float:
    """Inject a known weak tone into the target, cancel, and return recovered/injected amplitude.

    ~1.0 = the canceller preserves weak uncorrelated signals; << 1 = it is eating signal (UNSAFE).
    Recovery is measured differentially (with-pilot minus without-pilot lock-in) so leftover
    machine energy near f_pilot does not bias it.
    """
    t = np.arange(len(d)) / fs
    pilot = amp_g * np.sin(2 * np.pi * f_pilot * t)
    e0, _ = nlms_cancel(d, x, fs, cfg)
    e1, _ = nlms_cancel(d + pilot, x, fs, cfg)
    return float(abs(_lockin(e1, f_pilot, fs) - _lockin(e0, f_pilot, fs)) / amp_g)


def nlms_cancel_warm(d: np.ndarray, x: np.ndarray, fs: float, cfg: NLMSConfig = NLMSConfig(),
                     warm_s: float = 15.0) -> Tuple[np.ndarray, dict]:
    """Offline two-pass: learn weights on the first `warm_s` seconds, then run from there.

    A live system has a real convergence time (reported separately); for batch/dataset use this
    removes the start-up transient so it is not mistaken for a knock.
    """
    n = min(len(d), int(warm_s * fs))
    _, info0 = nlms_cancel(d[:n], x[:n], fs, cfg)
    return nlms_cancel(d, x, fs, cfg, w0=info0["w"])
