"""Run a scenario -> SimResult (streams + ground truth). Everything is SYNTHETIC."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from backend.core.types import RadarPacket, VibrationPacket
from simulator.body import motion_bursts_mm, respiration_mm, trajectory
from simulator.radar_model import simulate_camera, simulate_iq, simulate_tracks, tracks_to_packets
from simulator.scenario import ScenarioConfig
from simulator.vib_model import simulate_vibration

log = logging.getLogger(__name__)


@dataclass
class SimResult:
    cfg: ScenarioConfig
    iq_t: np.ndarray                 # (T,)
    iq: np.ndarray                   # (T, bins) complex64, Mode A stream
    range_axis: np.ndarray           # (bins,) metres
    tracks_t: np.ndarray             # (F,)
    tracks_arr: np.ndarray           # (F, 3, 4) int32, Mode B stream
    vib: np.ndarray                  # (N, n_sensors) g, includes 1 g DC and quantization
    vib_fs: float
    sensor_names: List[str]
    # ---- ground truth ----
    pids: List[int]
    kinds: List[str]
    buried: List[bool]
    person_pos: np.ndarray           # (P, T, 2) on the IQ time grid
    resp_mm: np.ndarray              # (P, T) true respiration waveform
    motion_mm: np.ndarray            # (P, T) true gross-motion waveform
    machine_active: np.ndarray       # (N,) bool on the vibration time grid
    vib_components: Dict[str, np.ndarray]   # machine / footsteps / knocks, each (N, n_sensors)
    knock_times: List[Tuple[int, float]]
    survivor_bins: List[int]         # true range bin of each buried person
    impairments: dict = field(default_factory=dict)
    cam_t: np.ndarray = None          # (Fc,) camera frame times
    cam_xy: np.ndarray = None         # (Fc, K, 2) world xy of visible people, NaN = none

    # ---- stream adapters ----
    def radar_packets(self, mode: str = "TRACKS") -> List[RadarPacket]:
        if mode == "TRACKS":
            return tracks_to_packets(self.tracks_t, self.tracks_arr, synthetic=True)
        if mode == "IQ":
            return [RadarPacket(t=float(t), iq=self.iq[k], synthetic=True) for k, t in enumerate(self.iq_t)]
        raise ValueError("mode must be TRACKS or IQ")

    def vibration_packets(self, frame_samples: int = 20) -> List[VibrationPacket]:
        out = []
        for seq, i in enumerate(range(0, len(self.vib) - frame_samples + 1, frame_samples)):
            out.append(VibrationPacket(i / self.vib_fs, self.vib_fs, self.vib[i:i + frame_samples].copy(),
                                       seq=seq & 0xFFFF, synthetic=True))
        return out

    # ---- persistence ----
    def save(self, path: str | Path) -> None:
        meta = dict(cfg=self.cfg.to_dict(), vib_fs=self.vib_fs, sensor_names=self.sensor_names, pids=self.pids,
                    kinds=self.kinds, buried=self.buried, knock_times=self.knock_times,
                    survivor_bins=self.survivor_bins, impairments=self.impairments, synthetic=True)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, meta=json.dumps(meta, default=float), iq_t=self.iq_t, iq=self.iq,
                            range_axis=self.range_axis, tracks_t=self.tracks_t, tracks_arr=self.tracks_arr,
                            vib=self.vib, person_pos=self.person_pos, resp_mm=self.resp_mm,
                            motion_mm=self.motion_mm, machine_active=self.machine_active,
                            cam_t=self.cam_t, cam_xy=self.cam_xy,
                            **{f"comp_{k}": v for k, v in self.vib_components.items()})

    @staticmethod
    def load(path: str | Path) -> "SimResult":
        z = np.load(path, allow_pickle=False)
        m = json.loads(str(z["meta"]))
        return SimResult(
            cfg=ScenarioConfig.from_dict(m["cfg"]), iq_t=z["iq_t"], iq=z["iq"], range_axis=z["range_axis"],
            tracks_t=z["tracks_t"], tracks_arr=z["tracks_arr"], vib=z["vib"], vib_fs=m["vib_fs"],
            sensor_names=m["sensor_names"], pids=m["pids"], kinds=m["kinds"], buried=m["buried"],
            person_pos=z["person_pos"], resp_mm=z["resp_mm"], motion_mm=z["motion_mm"],
            machine_active=z["machine_active"],
            vib_components={k: z[f"comp_{k}"] for k in ("machine", "footsteps", "knocks")},
            knock_times=[tuple(k) for k in m["knock_times"]], survivor_bins=m["survivor_bins"],
            impairments=m["impairments"], cam_t=z["cam_t"], cam_xy=z["cam_xy"])


def run_scenario(cfg: ScenarioConfig) -> SimResult:
    ss = np.random.SeedSequence(cfg.seed).spawn(5)
    rng_body, rng_radar, rng_trk, rng_vib, rng_cam = (np.random.default_rng(s) for s in ss)
    t = np.arange(0.0, cfg.duration_s, 1.0 / cfg.iq_fs_hz)

    resp, motion, disp = {}, {}, {}
    for p in cfg.persons:
        resp[p.pid] = respiration_mm(t, p.resp_hz, p.resp_amp_mm, rng_body)
        motion[p.pid], _, _ = motion_bursts_mm(t, p.motion_level, rng_body)
        disp[p.pid] = resp[p.pid] + motion[p.pid]

    iq, r_axis, imp = simulate_iq(cfg, t, disp, rng_radar)
    t_trk, arr = simulate_tracks(cfg, rng_trk)
    vib, comps, m_on, knocks = simulate_vibration(cfg, rng_vib)
    cam_t, cam_xy = simulate_camera(cfg, rng_cam)

    P = len(cfg.persons)
    pos = np.zeros((P, len(t), 2))
    for i, p in enumerate(cfg.persons):
        x, y, _, _ = trajectory(p, t)
        pos[i, :, 0], pos[i, :, 1] = x, y
    sbins = [int(np.argmin(np.abs(r_axis - np.hypot(p.x_m, p.y_m)))) for p in cfg.persons if p.buried]
    log.info("simulated %s seed=%d (SYNTHETIC): %d persons, %d machines", cfg.name, cfg.seed, P, len(cfg.machines))
    return SimResult(
        cfg=cfg, iq_t=t, iq=iq, range_axis=r_axis, tracks_t=t_trk, tracks_arr=arr, vib=vib, vib_fs=cfg.vib_fs_hz,
        sensor_names=[s.name for s in cfg.vib_sensors], pids=[p.pid for p in cfg.persons],
        kinds=[p.kind for p in cfg.persons], buried=[p.buried for p in cfg.persons], person_pos=pos,
        resp_mm=np.array([resp[p.pid] for p in cfg.persons]).reshape(P, len(t)),
        motion_mm=np.array([motion[p.pid] for p in cfg.persons]).reshape(P, len(t)),
        machine_active=m_on, vib_components=comps, knock_times=knocks, survivor_bins=sbins, impairments=imp, cam_t=cam_t, cam_xy=cam_xy)
