import numpy as np
import pandas as pd
import pytest
from scipy.signal import fftconvolve

from backend.core.types import EvidenceState
from backend.fusion.state_machine import StateConfig, StateMachine
from backend.models.baseline import Detector
from backend.models.features import ALL_COLS, features_from_sim
from backend.processing.adaptive import NLMSConfig, nlms_cancel, pilot_recovery
from backend.processing.phase import demodulate, fit_circle
from backend.processing.respiration import resp_features
from backend.processing.vibration import KNOCK_BANK, VibConfig, _template, detect_transients
from backend.tracking.exclusion import exclusion_mask
from simulator.scenario import make_scenario
from simulator.simulate import run_scenario

LAM = 0.0125


def _iq(disp_mm, fs=20.0, amp=0.1, offset=0.5 + 0.3j, noise=0.002, seed=0):
    rng = np.random.default_rng(seed)
    z = amp * np.exp(1j * 4 * np.pi * disp_mm / 1000 / LAM) + offset
    return z + noise * (rng.standard_normal(len(z)) + 1j * rng.standard_normal(len(z)))


def test_circle_fit_and_demod_recover_displacement():
    t = np.arange(0, 30, 1 / 20)
    d = 4.0 * np.sin(2 * np.pi * 0.27 * t)               # 4 mm => ~4 rad swing (non-linear regime)
    z = _iq(d)
    c, r = fit_circle(z)
    assert abs(c - (0.5 + 0.3j)) < 0.01 and abs(r - 0.1) < 0.01
    est, q = demodulate(z, LAM)
    assert np.corrcoef(est, d)[0, 1] > 0.99 and q["fit_residual"] < 0.1


def test_resp_features_rate_and_large_motion_gate():
    t = np.arange(0, 30, 1 / 20)
    d = 4.0 * np.sin(2 * np.pi * 0.27 * t)
    f = resp_features(_iq(d), 20.0, LAM, 0.002)
    assert f["valid"] == 1.0 and abs(f["rate_hz"] - 0.27) < 0.04 and f["peak_snr_db"] > 15
    big = d + 45 * np.sin(2 * np.pi * 0.08 * t)           # slow large body sway
    assert resp_features(_iq(big), 20.0, LAM, 0.002)["valid"] == 0.0


def test_nlms_cancels_machine_but_keeps_weak_signal():
    fs, n = 400.0, 400 * 60
    rng = np.random.default_rng(1)
    t = np.arange(n) / fs
    src = np.sin(2 * np.pi * 31 * t) + 0.5 * np.sin(2 * np.pi * 62 * t + 1) + 0.2 * rng.standard_normal(n)
    path = np.array([0.0, 0.0, 0.8, 0.3, -0.2])
    x = src + 0.01 * rng.standard_normal(n)
    d = 0.05 * fftconvolve(src, path)[:n] + 0.004 * rng.standard_normal(n)
    e, info = nlms_cancel(d, x, fs, NLMSConfig())
    rms = lambda a: np.sqrt(np.mean(a[n // 2:] ** 2))
    assert 20 * np.log10(rms(d) / rms(e)) > 10
    assert pilot_recovery(d[:8000], x[:8000], fs) > 0.8   # weak uncorrelated tone survives


def test_matched_filter_finds_knocks():
    fs = 400.0
    rng = np.random.default_rng(2)
    x = 0.004 * rng.standard_normal(int(20 * fs))
    h = _template(fs, 60.0, 0.03) * 0.05
    times = [5.0, 5.5, 6.0]
    for t0 in times:
        i = int(t0 * fs)
        x[i:i + len(h)] += h
    ev = [e for e in detect_transients(x, fs, VibConfig()) if e["kind"] == "knock"]
    assert all(any(abs(e["t"] - t) < 0.1 for e in ev) for t in times)


def test_exclusion_stationary_vs_walking():
    r_axis = (np.arange(64) + 0.5) * 0.1
    cam_t = np.arange(0, 60, 0.5)
    still = np.full((len(cam_t), 1, 2), np.nan)
    still[:, 0, 0], still[:, 0, 1] = 0.0, 3.0
    m = exclusion_mask(cam_t, still, 0, 30, r_axis)
    assert m[np.argmin(abs(r_axis - 3.0))] and not m[np.argmin(abs(r_axis - 4.5))]
    walk = np.full((len(cam_t), 1, 2), np.nan)
    walk[:, 0, 0], walk[:, 0, 1] = 0.0, np.linspace(1.5, 5.0, len(cam_t))   # crosses each bin only briefly
    mw = exclusion_mask(cam_t, walk, 0, 30, r_axis)
    assert mw.mean() < 0.9 and not mw[np.argmin(abs(r_axis - 5.0))]


def test_state_machine_rules():
    sm = StateMachine(StateConfig())
    assert sm.update(0.1) == EvidenceState.NO_EVIDENCE
    assert sm.update(0.6) == EvidenceState.UNCERTAIN            # needs persistence for POSSIBLE
    assert sm.update(0.6) == EvidenceState.POSSIBLE_SURVIVOR
    for _ in range(3):
        sm.update(0.95)
    assert sm.update(0.95) == EvidenceState.HIGH_CONFIDENCE_CANDIDATE
    assert sm.update(0.95, interference=0.99) == EvidenceState.POSSIBLE_SURVIVOR   # interference cap
    sm2 = StateMachine()
    for _ in range(6):
        s = sm2.update(0.95, quality=0.2)
    assert s == EvidenceState.UNCERTAIN                           # poor quality never exceeds UNCERTAIN


def test_features_from_streams_only():
    r = run_scenario(make_scenario("worker_survivor_machinery", 0, duration_s=60.0))
    df = features_from_sim(r)
    assert len(df) == 3 and set(ALL_COLS) <= set(df.columns)
    assert np.isfinite(df[ALL_COLS].to_numpy(float)).all()
    assert 0.0 <= df.pilot_recovery.iloc[0] <= 2.0


def test_detector_fit_predict_save_load(tmp_path):
    rng = np.random.default_rng(0)
    n = 300
    y = rng.integers(0, 2, n)
    df = pd.DataFrame({"a": y + rng.normal(0, 0.7, n), "b": rng.normal(size=n), "y_survivor": y})
    det = Detector(["a", "b"]).fit(df.iloc[:200], df.iloc[200:])
    p = det.predict_proba(df)
    assert p.min() >= 0 and p.max() <= 1 and np.corrcoef(p, y)[0, 1] > 0.3
    det.save(tmp_path / "m.joblib")
    assert np.allclose(Detector.load(tmp_path / "m.joblib").predict_proba(df), p)
