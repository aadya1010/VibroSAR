import numpy as np
import pytest
from dataclasses import replace
from scipy.signal import welch

from backend.sensors.sim_sources import SimRadarSource, SimVibrationSource
from simulator.labels import window_labels
from simulator.scenario import SCENARIO_NAMES, PersonSpec, make_scenario
from simulator.simulate import SimResult, run_scenario

CLEAN = dict(clutter_sigma=0.0, iq_dc_sigma=0.0, radar_noise=0.0005)


def test_ten_scenarios_exist_and_run_short():
    assert len(SCENARIO_NAMES) == 10
    for name in SCENARIO_NAMES:
        r = run_scenario(make_scenario(name, seed=1, duration_s=20.0))
        assert r.cfg.synthetic and r.iq.shape[0] == len(r.iq_t)
        assert r.vib.shape[1] == 2 and r.tracks_arr.shape[1:] == (3, 4)
        assert all(p.synthetic for p in r.radar_packets("TRACKS")[:5])
        assert all(p.synthetic for p in r.vibration_packets()[:5])


def test_deterministic_given_seed():
    a = run_scenario(make_scenario("worker_survivor_machinery", 3, duration_s=15))
    b = run_scenario(make_scenario("worker_survivor_machinery", 3, duration_s=15))
    c = run_scenario(make_scenario("worker_survivor_machinery", 4, duration_s=15))
    assert np.array_equal(a.iq, b.iq) and np.array_equal(a.vib, b.vib)
    assert not np.array_equal(a.vib, c.vib)


def test_survivor_respiration_visible_in_unwrapped_phase():
    cfg = make_scenario("survivor_alone", 0, duration_s=120.0, **CLEAN)
    r = run_scenario(cfg)
    p = cfg.persons[0]
    sig = r.iq[:, r.survivor_bins[0]]
    ph = np.unwrap(np.angle(sig))
    f, pxx = welch(ph - ph.mean(), fs=cfg.iq_fs_hz, nperseg=1024)
    band = (f > 0.1) & (f < 0.6)
    assert abs(f[band][np.argmax(pxx[band])] - p.resp_hz) < 0.04


def test_rubble_attenuates_radar_signal():
    base = make_scenario("survivor_alone", 0, duration_s=40.0, **CLEAN)
    p = base.persons[0]
    deep = replace(base, persons=[replace(p, depth_m=1.0)])
    shallow = replace(base, persons=[replace(p, depth_m=0.2)])
    a_deep = np.abs(run_scenario(deep).iq[:, run_scenario(deep).survivor_bins[0]]).mean()
    a_shallow = np.abs(run_scenario(shallow).iq[:, run_scenario(shallow).survivor_bins[0]]).mean()
    assert a_deep < 0.5 * a_shallow


def test_mode_b_rarely_sees_buried_survivor():
    cfg = make_scenario("worker_survivor", 0, duration_s=120.0, rubble_atten_db_per_m=30.0)
    r = run_scenario(cfg)
    wp, sp = cfg.persons[0], cfg.persons[1]

    def hit_rate(p):
        hits = 0
        for row in r.tracks_arr:
            live = row[row.any(axis=1)]  # skip empty slots (all zeros)
            hits += bool(len(live) and np.any((np.abs(live[:, 0] - p.x_m * 1000) < 400)
                                              & (np.abs(live[:, 1] - p.y_m * 1000) < 400)))
        return hits / len(r.tracks_arr)

    assert r.tracks_arr.any()
    assert hit_rate(sp) < 0.15                 # buried: almost never reported (24 GHz assumption)
    assert hit_rate(sp) < hit_rate(wp) or hit_rate(wp) < 0.3


def test_machine_harmonic_in_reference_sensor():
    r = run_scenario(make_scenario("worker_machinery", 2, duration_s=60.0))
    f0 = r.cfg.machines[0].f0_hz
    x = r.vib_components["machine"][:, 1]
    f, pxx = welch(x, fs=r.vib_fs, nperseg=2048)
    pk = f[np.argmax(pxx)]
    assert min(abs(pk - k * f0) / (k * f0) for k in range(1, 7)) < 0.06


def test_knocks_only_for_buried_survivor_and_labels():
    r = run_scenario(make_scenario("survivor_machinery", 0, duration_s=60.0))
    assert len(r.knock_times) >= 3 and np.abs(r.vib_components["knocks"]).max() > 0
    lab = window_labels(r)
    assert lab.survivor_buried.all() and lab.machine_active_frac.max() > 0.9 and lab.knock_in_window.any()
    none = run_scenario(make_scenario("one_worker", 0, duration_s=30.0))
    assert not none.knock_times and none.survivor_bins == []


def test_save_load_roundtrip(tmp_path):
    r = run_scenario(make_scenario("worker_survivor", 5, duration_s=20.0))
    r.save(tmp_path / "x.npz")
    r2 = SimResult.load(tmp_path / "x.npz")
    assert np.array_equal(r.iq, r2.iq) and r2.cfg.synthetic and r2.cfg.name == "worker_survivor"
    assert r2.knock_times == r.knock_times


def test_sim_sources_follow_interfaces():
    r = run_scenario(make_scenario("worker_survivor", 0, duration_s=10.0))
    s = SimRadarSource(r, "TRACKS")
    assert not s.capabilities().has_phase and s.capabilities().synthetic
    n = 0
    while not s.finished:
        n += len(s.read())
    assert n == len(r.tracks_t)
    assert SimRadarSource(r, "IQ").capabilities().has_phase
    v = SimVibrationSource(r)
    assert v.read()[0].n_channels == 2
