"""Vibration features: transient (knock vs footstep) analysis, machinery line-spectrum score,
cancellation metrics. Sampling rate fs (400 Hz in the ADXL345 setup); useful band 1-190 Hz.

Knock/footstep separation here is by dominant frequency and timing pattern. The
frequency split (footstep ~<40 Hz vs knock ~>40 Hz) comes from the SIMULATOR's
assumptions; it MUST be re-derived from real recordings before being trusted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from scipy.signal import coherence, find_peaks, welch

from backend.processing.filters import bandpass_sos, highpass_sos, zero_phase


@dataclass(frozen=True)
class VibConfig:
    hp_hz: float = 1.0
    env_lo: float = 15.0
    env_hi: float = 120.0
    env_smooth_s: float = 0.02
    thr_mads: float = 6.0           # matched-filter score threshold (robust z)
    min_gap_s: float = 0.15
    knock_f_lo: float = 40.0        # SIMULATOR assumption, recalibrate on real data
    knock_f_hi: float = 100.0
    step_f_lo: float = 10.0
    step_f_hi: float = 40.0
    triplet_span_s: float = 3.0
    knock_gap_lo: float = 0.3
    knock_gap_hi: float = 0.8
    interf_scale_g: float = 0.01
    noise_floor_g: float = 0.005    # sensor noise floor; interference = excess reference energy above it


def preprocess(x: np.ndarray, fs: float, cfg: VibConfig = VibConfig()) -> np.ndarray:
    """Remove gravity/DC and drift (high-pass)."""
    return zero_phase(x - np.mean(x), highpass_sos(cfg.hp_hz, fs))


def _template(fs: float, f_hz: float, tau_s: float) -> np.ndarray:
    t = np.arange(int(6 * tau_s * fs)) / fs
    h = np.exp(-t / tau_s) * np.sin(2 * np.pi * f_hz * t)
    return h / np.linalg.norm(h)


# Template banks are SIMULATOR assumptions (damped sinusoids). Replace by templates learned from real taps/steps.
KNOCK_BANK = [(f, 0.03) for f in (45.0, 60.0, 75.0, 90.0)]
STEP_BANK = [(f, 0.04) for f in (15.0, 25.0, 35.0)]


def detect_transients(x: np.ndarray, fs: float, cfg: VibConfig = VibConfig()) -> List[dict]:
    """Matched-filter detection. Each event: t, z (robust SNR score), f_dom (best template freq), kind."""
    from scipy.signal import fftconvolve
    best, kind, fbest = None, None, None
    for name, bank in (("knock", KNOCK_BANK), ("step", STEP_BANK)):
        for f, tau in bank:
            h = _template(fs, f, tau)
            m = np.abs(fftconvolve(x, h[::-1], mode="same"))
            med = np.median(m)
            mad = np.median(np.abs(m - med)) * 1.4826 + 1e-12
            z = (m - med) / mad
            if best is None:
                best, kind, fbest = z.copy(), np.full(len(z), name, dtype=object), np.full(len(z), f)
            else:
                up = z > best
                best[up], kind[up], fbest[up] = z[up], name, f
    pk, _ = find_peaks(best, height=cfg.thr_mads, distance=int(cfg.min_gap_s * fs))
    return [dict(t=float(i / fs), z=float(best[i]), amp=float(best[i]), f_dom=float(fbest[i]), kind=str(kind[i])) for i in pk]


def _triplet_score(times: np.ndarray, cfg: VibConfig) -> float:
    """1.0 if >=3 knock-like events fall within triplet_span_s with plausible gaps, scaled by count."""
    best = 0
    for i, t0 in enumerate(times):
        grp = times[(times >= t0) & (times < t0 + cfg.triplet_span_s)]
        gaps = np.diff(grp)
        if len(grp) >= 2 and np.all((gaps >= cfg.knock_gap_lo) & (gaps <= cfg.knock_gap_hi)):
            best = max(best, len(grp))
    return float(min(best, 3) / 3.0)


def vib_window_features(clean: np.ndarray, raw: np.ndarray, ref: np.ndarray, fs: float,
                        cfg: VibConfig = VibConfig()) -> Dict[str, float]:
    """All inputs: 1-D, high-passed, same window. `clean` = canceller output on the target."""
    tr = detect_transients(clean, fs, cfg)
    f = np.array([t["f_dom"] for t in tr])
    ts = np.array([t["t"] for t in tr])
    kinds = np.array([t["kind"] for t in tr])
    knock = ts[kinds == "knock"] if len(tr) else np.array([])
    step = ts[kinds == "step"] if len(tr) else np.array([])
    fr, pr = welch(raw, fs=fs, nperseg=min(512, len(raw)))
    _, pc = welch(clean, fs=fs, nperseg=min(512, len(clean)))
    band = (fr >= 20) & (fr <= 190)
    peaky = float(np.max(pr[band]) / (np.median(pr[band]) + 1e-18))
    fc, cxy = coherence(raw, ref, fs=fs, nperseg=min(256, len(raw)))
    cb = (fc >= 10) & (fc <= 150)
    rms = lambda a: float(np.sqrt(np.mean(a ** 2)))
    out = dict(
        raw_rms_g=rms(raw), clean_rms_g=rms(clean), ref_rms_g=rms(ref),
        cancel_gain_db=float(10 * np.log10((np.sum(pr[band]) + 1e-18) / (np.sum(pc[band]) + 1e-18))),
        machine_peakiness=np.log10(peaky), coh_raw_ref=float(np.mean(cxy[cb])),
        interference=float(1 - np.exp(-max(rms(ref) - cfg.noise_floor_g, 0.0) / cfg.interf_scale_g)),
        n_transients=float(len(tr)), n_knock_like=float(len(knock)), n_step_like=float(len(step)),
        knock_triplet_score=_triplet_score(knock, cfg),
        max_knock_z=float(max([t["z"] for t in tr if t["kind"] == "knock"], default=0.0)),
        step_rate_hz=float(len(step) / (len(clean) / fs)),
    )
    for lo, hi in ((1, 10), (10, 30), (30, 80), (80, 190)):
        m = (fr >= lo) & (fr < hi)
        out[f"clean_band_{lo}_{hi}"] = float(np.log10(np.sum(pc[m]) + 1e-18))
    return out
