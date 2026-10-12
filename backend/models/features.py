"""Window feature extraction from raw streams (NO ground truth is read here).

Feature groups:
  RESP  - radar-phase respiration evidence (needs Mode A raw phase; NOT available on RD-03D)
  EXCL  - camera exclusion context
  TRK   - processed radar-track context (Mode B, available on RD-03D)
  VIB   - vibration evidence after adaptive cancellation (ADXL345)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from backend.processing.adaptive import NLMSConfig, nlms_cancel_warm, pilot_recovery
from backend.processing.respiration import RespConfig, resp_analyze
from backend.processing.vibration import VibConfig, preprocess, vib_window_features
from backend.tracking.exclusion import exclusion_mask


@dataclass(frozen=True)
class FeatureConfig:
    win_s: float = 30.0
    hop_s: float = 10.0
    active_std_k: float = 3.0         # bin is 'active' if temporal std > k x median bin std
    snr_cand_db: float = 10.0
    trk_persist_hits: int = 5
    margin_m: float = 0.35
    ghost_lo_m: float = -1.0           # blanket ghost zone OFF; ghosts are rejected by waveform correlation
    ghost_hi_m: float = -1.0
    dup_corr: float = 0.7              # candidate whose waveform correlates >= this with a visible person's = ghost/duplicate
    resp: RespConfig = field(default_factory=RespConfig)
    vib: VibConfig = field(default_factory=VibConfig)
    nlms: NLMSConfig = field(default_factory=NLMSConfig)


RESP_COLS = ["resp_snr_db", "resp_rate_hz", "resp_amp_mm", "resp_entropy", "resp_periodicity",
             "resp_harmonic", "resp_stability_hz", "resp_motion_frac", "n_resp_cands", "n_resp_cands_excl", "n_resp_ghost"]
EXCL_COLS = ["n_cam_hits", "n_excl_bins"]
TRK_COLS = ["trk_frame_frac", "trk_mean_count", "trk_persist_bins"]
VIB_COLS = ["raw_rms_g", "clean_rms_g", "ref_rms_g", "cancel_gain_db", "machine_peakiness", "coh_raw_ref",
            "interference", "n_transients", "n_knock_like", "n_step_like", "knock_triplet_score", "max_knock_z",
            "step_rate_hz", "pilot_recovery", "clean_band_1_10", "clean_band_10_30", "clean_band_30_80",
            "clean_band_80_190"]
ALL_COLS = RESP_COLS + EXCL_COLS + TRK_COLS + VIB_COLS


def estimate_noise_sigma(iq: np.ndarray) -> float:
    """Per-component noise sigma from sample-to-sample differences (robust across bins; no truth used)."""
    d = np.diff(iq, axis=0)
    s = 0.5 * (np.median(np.std(d.real, axis=0)) + np.median(np.std(d.imag, axis=0))) / np.sqrt(2)
    return float(s)


def extract_window_features(*, iq: np.ndarray, iq_t: np.ndarray, iq_fs: float, r_axis: np.ndarray,
                            wavelength_m: float, trk_t: np.ndarray, trk_arr: np.ndarray,
                            cam_t: np.ndarray, cam_xy: np.ndarray, vib: np.ndarray, vib_fs: float,
                            cfg: FeatureConfig = FeatureConfig()) -> pd.DataFrame:
    sigma = estimate_noise_sigma(iq)
    # ---- vibration: whole-recording cancellation (target=ch0, reference=ch1) ----
    d = preprocess(vib[:, 0], vib_fs, cfg.vib)
    x = preprocess(vib[:, 1], vib_fs, cfg.vib)
    clean, _ = nlms_cancel_warm(d, x, vib_fs, cfg.nlms)
    # pilot test on the 10 s segment where the reference is strongest
    seg = int(10 * vib_fs)
    starts = np.arange(0, len(d) - seg, seg // 2)
    i0 = int(starts[np.argmax([np.sum(x[s:s + seg] ** 2) for s in starts])])
    pil = pilot_recovery(d[i0:i0 + seg], x[i0:i0 + seg], vib_fs, cfg.nlms)

    rows: List[Dict[str, float]] = []
    T = iq_t[-1]
    t0 = 0.0
    while t0 + cfg.win_s <= T + 1e-6:
        t1 = t0 + cfg.win_s
        ia, ib = np.searchsorted(iq_t, [t0, t1])
        seg_iq = iq[ia:ib]
        excl = exclusion_mask(cam_t, cam_xy, t0, t1, r_axis, cfg.margin_m, cfg.ghost_lo_m, cfg.ghost_hi_m)
        sd = np.std(seg_iq, axis=0)
        active = sd > cfg.active_std_k * np.median(sd)
        cands = []
        for b in np.flatnonzero(active):
            f, wave = resp_analyze(seg_iq[:, b].astype(complex), iq_fs, wavelength_m, sigma, cfg.resp)
            if f["valid"] and f["peak_snr_db"] >= cfg.snr_cand_db:
                cands.append((b, f, wave))
        vis = [c for c in cands if excl[c[0]]]                      # inside a visible person's range zone
        free = [c for c in cands if not excl[c[0]]]
        keep = []
        for b, f, w in free:                                        # ghost/duplicate rejection by waveform correlation
            dup = any(abs(np.corrcoef(w, vw)[0, 1]) >= cfg.dup_corr for _, _, vw in vis)
            if not dup:
                keep.append((b, f))
        n_ghost = len(free) - len(keep)
        row: Dict[str, float] = dict(t0=t0, t1=t1)
        if keep:
            b, f = max(keep, key=lambda bf: bf[1]["peak_snr_db"])
            row.update(resp_snr_db=f["peak_snr_db"], resp_rate_hz=f["rate_hz"], resp_amp_mm=f["amp_mm_rms"],
                       resp_entropy=f["entropy"], resp_periodicity=f["periodicity"], resp_harmonic=f["harmonic_ratio"],
                       resp_stability_hz=f["rate_stability_hz"], resp_motion_frac=f["motion_frac"],
                       best_bin=int(b), best_range_m=float(r_axis[b]))
        else:
            row.update(resp_snr_db=-30.0, resp_rate_hz=0.0, resp_amp_mm=0.0, resp_entropy=1.0, resp_periodicity=0.0,
                       resp_harmonic=0.0, resp_stability_hz=1.0, resp_motion_frac=1.0, best_bin=-1, best_range_m=-1.0)
        row.update(n_resp_cands=float(len(keep)), n_resp_cands_excl=float(len(vis)), n_resp_ghost=float(n_ghost),
                   naive_flag=float(len(cands) > 0))  # 'breathing anywhere' detector, no exclusion
        # camera context
        sel = (cam_t >= t0) & (cam_t <= t1)
        xy = cam_xy[sel].reshape(-1, 2)
        row.update(n_cam_hits=float((~np.isnan(xy).any(axis=1)).sum()), n_excl_bins=float(excl.sum()))
        # radar-track context (Mode B)
        tsel = (trk_t >= t0) & (trk_t <= t1)
        ta = trk_arr[tsel]
        live = ta.any(axis=2)                        # (F, 3)
        rng_m = np.hypot(ta[:, :, 0], ta[:, :, 1])[live] / 1000.0
        hist = np.histogram(rng_m, bins=np.append(r_axis - 0.05, r_axis[-1] + 0.05))[0] if len(rng_m) else np.zeros(len(r_axis))
        row.update(trk_frame_frac=float(live.any(axis=1).mean()) if len(ta) else 0.0,
                   trk_mean_count=float(live.sum(axis=1).mean()) if len(ta) else 0.0,
                   trk_persist_bins=float((hist >= cfg.trk_persist_hits).sum()))
        # vibration
        a, b_ = int(t0 * vib_fs), int(t1 * vib_fs)
        v = vib_window_features(clean[a:b_], d[a:b_], x[a:b_], vib_fs, cfg.vib)
        row.update(v)
        row["pilot_recovery"] = pil
        rows.append(row)
        t0 += cfg.hop_s
    return pd.DataFrame(rows)


def features_from_sim(r, cfg: FeatureConfig = FeatureConfig()) -> pd.DataFrame:
    """Feed ONLY stream arrays (never truth) from a SimResult into the extractor."""
    return extract_window_features(iq=r.iq, iq_t=r.iq_t, iq_fs=r.cfg.iq_fs_hz, r_axis=r.range_axis,
                                   wavelength_m=r.cfg.wavelength_m, trk_t=r.tracks_t, trk_arr=r.tracks_arr,
                                   cam_t=r.cam_t, cam_xy=r.cam_xy, vib=r.vib, vib_fs=r.vib_fs, cfg=cfg)
