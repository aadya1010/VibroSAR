"""Synthetic radar models.

Mode A (IQ): output is the slow-time x range-bin complex matrix AFTER range
processing (i.e. what a raw-phase radar would give per chirp-frame), not raw
ADC samples. Phase = 4*pi*R/lambda. At 24 GHz (lambda=12.5 mm) a 4 mm chest
motion swings the phase by ~4 rad, so the signal is strongly non-linear
(harmonics, wrapping): the DSP stage must use arctangent demodulation with
circle-fit DC removal, not naive mean subtraction.

Mode B (TRACKS): an EMULATION of a processed-target radar (RD-03D class):
~10 Hz, up to 3 targets, mostly sees MOVING things, drops stationary ones.
Detection probabilities are assumptions, not device measurements.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from backend.core.types import RadarPacket, RadarTrack
from simulator.body import trajectory
from simulator.scenario import MachineSpec, PersonSpec, ScenarioConfig


def _add_to_bins(iq: np.ndarray, sig: np.ndarray, R: np.ndarray, r_axis: np.ndarray, bin_m: float) -> None:
    w = np.exp(-0.5 * ((r_axis[None, :] - R[:, None]) / (0.7 * bin_m)) ** 2)
    w[w < 0.01] = 0.0
    iq += sig[:, None] * w


def machine_active_mask(m: MachineSpec, t: np.ndarray) -> np.ndarray:
    t_off = m.t_off_s if m.t_off_s is not None else float(t[-1]) + 1.0
    return np.clip((t - m.t_on_s) / 0.5, 0, 1) * np.clip((t_off - t) / 0.5, 0, 1)


def simulate_iq(cfg: ScenarioConfig, t: np.ndarray, disp_mm: Dict[int, np.ndarray],
                rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, dict]:
    r_axis = (np.arange(cfg.n_bins) + 0.5) * cfg.bin_m
    T = len(t)
    iq = np.zeros((T, cfg.n_bins), dtype=np.complex128)
    iq += (rng.standard_normal(cfg.n_bins) + 1j * rng.standard_normal(cfg.n_bins))[None, :] * cfg.clutter_sigma

    for p in cfg.persons:
        x, y, _, _ = trajectory(p, t)
        R = np.hypot(x, y) + disp_mm[p.pid] / 1000.0
        amp = p.rcs / np.maximum(R, 0.3) ** 2
        if p.buried:
            amp = amp * 10 ** (-cfg.rubble_atten_db_per_m * p.depth_m / 20.0)
        _add_to_bins(iq, amp * np.exp(1j * 4 * np.pi * R / cfg.wavelength_m), R, r_axis, cfg.bin_m)
        if not p.buried:  # multipath ghost behind visible people: mimics a source in an occluded region
            Rg = R + cfg.ghost_extra_m
            _add_to_bins(iq, cfg.ghost_gain * amp * np.exp(1j * 4 * np.pi * Rg / cfg.wavelength_m),
                         Rg, r_axis, cfg.bin_m)

    for m in cfg.machines:  # machine body vibration, sampled at the slow rate => it ALIASES (intended)
        Rm = float(np.hypot(m.x_m, m.y_m))
        act = machine_active_mask(m, t)
        d = 0.2e-3 * sum(np.sin(2 * np.pi * k * m.f0_hz * t) / k for k in (1, 2, 3)) * act * (m.strength_g / 0.2)
        _add_to_bins(iq, cfg.machine_radar_amp * np.exp(1j * 4 * np.pi * (Rm + d) / cfg.wavelength_m),
                     np.full(T, Rm), r_axis, cfg.bin_m)

    iq += cfg.radar_noise * (rng.standard_normal(iq.shape) + 1j * rng.standard_normal(iq.shape))

    # front-end impairments (gain/phase imbalance, DC offset): constant per recording
    g, eps = rng.uniform(0.95, 1.08), np.deg2rad(rng.uniform(-4, 4))
    dc = rng.normal(0, cfg.iq_dc_sigma, 2)
    i_ch = iq.real + dc[0]
    q_ch = g * (iq.imag * np.cos(eps) + iq.real * np.sin(eps)) + dc[1]
    return (i_ch + 1j * q_ch).astype(np.complex64), r_axis, {"gain": g, "phase_err_rad": eps, "dc": dc.tolist()}


def simulate_tracks(cfg: ScenarioConfig, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Return (t_frames, arr) with arr shape (F, 3, 4) int32 [x_mm, y_mm, speed_cm_s, res_mm]; all-zero row = empty."""
    t = np.arange(0.0, cfg.duration_s, 1.0 / cfg.track_fs_hz)
    arr = np.zeros((len(t), 3, 4), dtype=np.int32)
    paths = {p.pid: trajectory(p, t) for p in cfg.persons}
    for k in range(len(t)):
        found = []
        for p in cfg.persons:
            x, y, vx, vy = (a[k] for a in paths[p.pid])
            sp = float(np.hypot(vx, vy))
            pd = np.clip(0.15 + 0.7 * min(1.0, sp / 0.3) + 0.3 * p.motion_level, 0.0, 0.98)
            if p.buried:
                pd *= 10 ** (-cfg.rubble_atten_db_per_m * p.depth_m / 20.0)
            if rng.random() < pd:
                R = max(float(np.hypot(x, y)), 1e-3)
                v_rad = (x * vx + y * vy) / R
                found.append((R, int(round((x + rng.normal(0, 0.04)) * 1000)),
                              int(round((y + rng.normal(0, 0.04)) * 1000)),
                              int(round((v_rad + rng.normal(0, 0.03)) * 100)), 360))
        found.sort()
        for j, (_, xm, ym, v, res) in enumerate(found[:3]):
            arr[k, j] = (xm, ym, v, res)
    return t, arr


def tracks_to_packets(t: np.ndarray, arr: np.ndarray, synthetic: bool = True) -> List[RadarPacket]:
    out = []
    for k in range(len(t)):
        tr = tuple(RadarTrack(int(r[0]), int(r[1]), int(r[2]), int(r[3])) for r in arr[k] if r.any())
        out.append(RadarPacket(t=float(t[k]), tracks=tr, synthetic=synthetic))
    return out


def simulate_camera(cfg: ScenarioConfig, rng: np.random.Generator, fs: float = 2.0,
                    p_detect: float = 0.9, sigma_m: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
    """Emulated person detector after ground-plane calibration: world (x, y) of VISIBLE people only.

    Buried people are never detected (occluded). Output xy shape (F, max(1,P), 2), NaN = no detection.
    Detection probability and position noise are assumptions.
    """
    t = np.arange(0.0, cfg.duration_s, 1.0 / fs)
    K = max(1, len(cfg.persons))
    xy = np.full((len(t), K, 2), np.nan)
    for j, p in enumerate(cfg.persons):
        if p.buried:
            continue
        x, y, _, _ = trajectory(p, t)
        seen = rng.random(len(t)) < p_detect
        xy[seen, j, 0] = x[seen] + rng.normal(0, sigma_m, seen.sum())
        xy[seen, j, 1] = y[seen] + rng.normal(0, sigma_m, seen.sum())
    return t, xy
