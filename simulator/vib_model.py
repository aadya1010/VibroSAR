"""Synthetic vibration (ADXL345-like, units of g). All components are SYNTHETIC.

Components: machinery (harmonic comb + AM + wideband, non-stationary), worker
footsteps, survivor knocks. Each source reaches each sensor through a short
multipath impulse response (delay + attenuation + echoes). Components are kept
separately so noise-cancellation can be scored against the true machine part.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import butter, sosfiltfilt

from simulator.body import lowpass_noise, motion_bursts_mm, trajectory
from simulator.radar_model import machine_active_mask
from simulator.scenario import MachineSpec, ScenarioConfig


def _taps(rng: np.random.Generator, horiz_m: float, depth_m: float, cfg: ScenarioConfig) -> List[Tuple[float, float]]:
    path = float(np.hypot(horiz_m, depth_m))
    att_db = cfg.vib_atten_db_per_m * horiz_m + cfg.vib_rubble_atten_db_per_m * depth_m
    g0 = 10 ** (-att_db / 20.0) / max(1.0, path)
    taps, delay = [(path / cfg.wave_speed_m_s, g0)], path / cfg.wave_speed_m_s
    for k in range(1, 4):
        delay += rng.uniform(0.003, 0.03)
        taps.append((delay, g0 * rng.choice([-1, 1]) * rng.uniform(0.2, 0.6) ** k))
    return taps


def _apply(sig: np.ndarray, taps: List[Tuple[float, float]], fs: float) -> np.ndarray:
    out = np.zeros_like(sig)
    for d, g in taps:
        s = int(round(d * fs))
        if s < len(sig):
            out[s:] += g * sig[: len(sig) - s]
    return out


def _damped(n: int, fs: float, times: List[float], f_hz: float, tau_s: float, amp: float) -> np.ndarray:
    sig, L = np.zeros(n), int(6 * tau_s * fs)
    tt = np.arange(L) / fs
    pulse = amp * np.exp(-tt / tau_s) * np.sin(2 * np.pi * f_hz * tt)
    for t0 in times:
        i = int(t0 * fs)
        if 0 <= i < n:
            m = min(L, n - i)
            sig[i:i + m] += pulse[:m]
    return sig


def _machine_source(m: MachineSpec, t: np.ndarray, fs: float, rng: np.random.Generator) -> np.ndarray:
    n = len(t)
    f = m.f0_hz * (1.0 + 0.03 * lowpass_noise(rng, n, fs, 0.1))          # drifting speed
    phi = 2 * np.pi * np.cumsum(f) / fs
    am = 1.0 + 0.3 * np.sin(2 * np.pi * 0.3 * t + rng.uniform(0, 6.28))   # load modulation
    comb = sum((1.0 / k ** 0.8) * np.sin(k * phi + rng.uniform(0, 6.28)) for k in range(1, 7) if k * m.f0_hz < 0.45 * fs)
    sos = butter(2, [20, min(180, 0.45 * fs)], btype="band", fs=fs, output="sos")
    wide = sosfiltfilt(sos, rng.standard_normal(n))
    s = am * comb + 0.3 * wide / (wide.std() + 1e-12)
    s = s / (s.std() + 1e-12) * m.strength_g
    return s * machine_active_mask(m, t)


def simulate_vibration(cfg: ScenarioConfig, rng: np.random.Generator
                       ) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray, List[Tuple[int, float]]]:
    fs = cfg.vib_fs_hz
    t = np.arange(0.0, cfg.duration_s, 1.0 / fs)
    n, S = len(t), len(cfg.vib_sensors)
    comp = {k: np.zeros((n, S)) for k in ("machine", "footsteps", "knocks")}
    machine_on = np.zeros(n, bool)
    knock_times: List[Tuple[int, float]] = []

    for m in cfg.machines:
        src = _machine_source(m, t, fs, rng)
        machine_on |= machine_active_mask(m, t) > 0.5
        for j, s in enumerate(cfg.vib_sensors):
            comp["machine"][:, j] += _apply(src, _taps(rng, float(np.hypot(m.x_m - s.x_m, m.y_m - s.y_m)), 0.0, cfg), fs)

    tp = np.arange(0.0, cfg.duration_s, 0.05)
    for p in cfg.persons:
        if not p.buried:  # footsteps / posture-shift impacts
            x, y, vx, vy = trajectory(p, tp)
            speed = float(np.mean(np.hypot(vx, vy)))
            rate = 1.8 * min(1.0, speed) + 0.1 * p.motion_level
            n_ev = rng.poisson(rate * cfg.duration_s)
            times = list(rng.uniform(0, cfg.duration_s, n_ev))
            src = _damped(n, fs, times, 25.0, 0.04, cfg.footstep_g)
            px, py = float(np.mean(x)), float(np.mean(y))
            for j, s in enumerate(cfg.vib_sensors):
                comp["footsteps"][:, j] += _apply(src, _taps(rng, float(np.hypot(px - s.x_m, py - s.y_m)), 0.0, cfg), fs)
        elif p.knocks:  # bursts of 3 knocks, ~0.5 s apart, roughly every knock_interval_s
            times, tb = [], rng.uniform(2, cfg.knock_interval_s)
            while tb < cfg.duration_s - 2:
                times += [tb, tb + 0.5, tb + 1.0]
                knock_times += [(p.pid, tb), (p.pid, tb + 0.5), (p.pid, tb + 1.0)]
                tb += cfg.knock_interval_s * rng.uniform(0.7, 1.3)
            src = _damped(n, fs, times, 60.0, 0.03, cfg.knock_g)
            for j, s in enumerate(cfg.vib_sensors):
                comp["knocks"][:, j] += _apply(src, _taps(rng, float(np.hypot(p.x_m - s.x_m, p.y_m - s.y_m)), p.depth_m, cfg), fs)

    total = sum(comp.values()) + cfg.vib_noise_g * rng.standard_normal((n, S))
    total[:, :] += cfg.dc_offset_g * np.array([1.0] * S)[None, :]  # gravity on the sensed axis
    total = np.clip(total, -2.0, 2.0)
    total = np.round(total / cfg.adc_lsb_g) * cfg.adc_lsb_g          # ADC quantization
    return total, comp, machine_on, knock_times
