"""Respiration-like micro-motion features from demodulated displacement (Mode A only).

Sampling: iq_fs (20 Hz in the simulator). Band: 0.1-0.6 Hz (6-36 breaths/min).
Latency: one analysis window (default 30 s) -> frequency resolution ~0.033 Hz.
This module separates SMALL periodic motion (breathing-like) from LARGE motion
(walking, arm/posture shifts) via a segment peak-to-peak gate; large-motion
windows are reported as invalid instead of being scored.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import detrend

from backend.processing.filters import bandpass_sos, zero_phase
from backend.processing.phase import demodulate, fit_circle


@dataclass(frozen=True)
class RespConfig:
    band_lo: float = 0.1
    band_hi: float = 0.6
    peak_lo: float = 0.12
    peak_hi: float = 0.55
    motion_seg_s: float = 4.0
    motion_p2p_mm: float = 20.0       # segment peak-to-peak above this = large motion
    motion_frac_max: float = 0.25     # window invalid if more segments than this are "large motion"
    min_radius_ratio: float = 2.0     # circle radius must exceed this x noise estimate
    amp_cv_max: float = 0.3           # amplitude non-stationarity gate (a target walking through a bin varies a lot)


def _peak_freq(x: np.ndarray, fs: float, cfg: RespConfig) -> float:
    n = 4096
    f = np.fft.rfftfreq(n, 1 / fs)
    p = np.abs(np.fft.rfft(x * np.hanning(len(x)), n)) ** 2
    m = (f >= cfg.peak_lo) & (f <= cfg.peak_hi)
    return float(f[m][np.argmax(p[m])])


def resp_features(z: np.ndarray, fs: float, wavelength_m: float, noise_sigma: float,
                  cfg: RespConfig = RespConfig()) -> Dict[str, float]:
    """z: complex slow-time samples of ONE range bin for ONE window."""
    return resp_analyze(z, fs, wavelength_m, noise_sigma, cfg)[0]


def resp_analyze(z: np.ndarray, fs: float, wavelength_m: float, noise_sigma: float,
                 cfg: RespConfig = RespConfig()) -> Tuple[Dict[str, float], Optional[np.ndarray]]:
    """Like resp_features but also returns the band-limited displacement waveform (None if invalid)."""
    disp, q = demodulate(z, wavelength_m)
    out: Dict[str, float] = dict(valid=0.0, rate_hz=np.nan, peak_snr_db=-30.0, amp_mm_rms=0.0, entropy=1.0,
                                 harmonic_ratio=0.0, periodicity=0.0, rate_stability_hz=1.0, motion_frac=1.0,
                                 fit_residual=q["fit_residual"], radius_ratio=q["radius"] / (noise_sigma + 1e-12), amp_cv=1.0)
    c, _ = fit_circle(z)
    a = uniform_filter1d(np.abs(z - c), max(1, int(3 * fs)))
    out["amp_cv"] = float(np.std(a) / (np.mean(a) + 1e-12))
    # large-motion gate on the raw (unfiltered) displacement
    seg = int(cfg.motion_seg_s * fs)
    dt = detrend(disp)
    p2p = [np.ptp(dt[i:i + seg]) for i in range(0, len(dt) - seg + 1, seg)]
    out["motion_frac"] = float(np.mean(np.array(p2p) > cfg.motion_p2p_mm)) if p2p else 1.0
    if (out["radius_ratio"] < cfg.min_radius_ratio or out["motion_frac"] > cfg.motion_frac_max
            or out["amp_cv"] > cfg.amp_cv_max):
        return out, None
    x = zero_phase(dt, bandpass_sos(cfg.band_lo, cfg.band_hi, fs))
    n = 4096
    f = np.fft.rfftfreq(n, 1 / fs)
    p = np.abs(np.fft.rfft(x * np.hanning(len(x)), n)) ** 2
    band = (f >= cfg.band_lo) & (f <= cfg.band_hi)
    pk = _peak_freq(x, fs, cfg)
    near = band & (np.abs(f - pk) <= 0.03)
    floor = np.median(p[band & ~near]) + 1e-18
    pb = p[band] / (p[band].sum() + 1e-18)
    h = f.copy()
    h2 = (np.abs(f - 2 * pk) <= 0.03)
    lag = int(round(fs / pk))
    ac = np.correlate(x - x.mean(), x - x.mean(), "full")[len(x) - 1:]
    # stability: peak frequency in three overlapping sub-windows
    L = len(x)
    subs = [x[i:i + int(0.6 * L)] for i in (0, int(0.2 * L), int(0.4 * L))]
    out.update(valid=1.0, rate_hz=pk,
               peak_snr_db=float(10 * np.log10(p[near].max() / floor)),
               amp_mm_rms=float(x.std()),
               entropy=float(-(pb * np.log(pb + 1e-18)).sum() / np.log(len(pb))),
               harmonic_ratio=float(p[h2].max() / (p[near].max() + 1e-18)) if h2.any() else 0.0,
               periodicity=float(ac[lag] / (ac[0] + 1e-18)) if lag < len(ac) else 0.0,
               rate_stability_hz=float(np.std([_peak_freq(s, fs, cfg) for s in subs])))
    return out, x
