"""Scenario description for the VibroSAR simulator.

EVERYTHING produced from these configs is SYNTHETIC. The models are physically
motivated (phase = 4*pi*R/lambda, exponential rubble attenuation, multipath
ghosts, propagation delays) but every parameter is an assumption to be replaced
by lab measurements. Never report simulator numbers as real-world performance.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class PersonSpec:
    pid: int
    kind: str                      # "worker" | "survivor"
    x_m: float
    y_m: float
    vx_m_s: float = 0.0
    vy_m_s: float = 0.0
    buried: bool = False
    depth_m: float = 0.0           # depth under rubble surface (only if buried)
    resp_hz: float = 0.27          # ~16 breaths/min
    resp_amp_mm: float = 4.0       # chest displacement amplitude seen by radar
    motion_level: float = 0.05     # 0..1, gross body motion intensity
    rcs: float = 0.8               # arbitrary radar cross-section scale
    knocks: bool = False           # survivor actively knocking (vibration evidence)

    def __post_init__(self) -> None:
        if self.kind not in ("worker", "survivor"):
            raise ValueError(f"kind must be worker|survivor, got {self.kind}")
        if self.buried and self.depth_m <= 0:
            raise ValueError("buried person needs depth_m > 0")
        if not 0.05 <= self.resp_hz <= 1.0:
            raise ValueError("resp_hz outside plausible 0.05-1.0 Hz")
        if not 0.0 <= self.motion_level <= 1.0:
            raise ValueError("motion_level must be in [0, 1]")


@dataclass
class MachineSpec:
    x_m: float
    y_m: float
    f0_hz: float = 30.0
    strength_g: float = 0.2        # RMS at 1 m
    t_on_s: float = 0.0
    t_off_s: Optional[float] = None  # None = until the end


@dataclass
class SensorSpec:
    name: str
    x_m: float
    y_m: float


@dataclass
class ScenarioConfig:
    name: str
    seed: int = 0
    duration_s: float = 120.0
    persons: List[PersonSpec] = field(default_factory=list)
    machines: List[MachineSpec] = field(default_factory=list)
    vib_sensors: List[SensorSpec] = field(default_factory=lambda: [
        SensorSpec("target", 0.5, 3.0), SensorSpec("reference", -2.6, 2.0)])
    # ---- radar (radar at origin, looking along +y) ----
    wavelength_m: float = 0.0125   # 24 GHz class
    iq_fs_hz: float = 20.0         # slow-time rate after range processing
    n_bins: int = 64
    bin_m: float = 0.1
    track_fs_hz: float = 10.0
    radar_noise: float = 0.004
    clutter_sigma: float = 0.3
    iq_dc_sigma: float = 0.05
    ghost_gain: float = 0.3        # multipath ghost of visible people
    ghost_extra_m: float = 0.8
    machine_radar_amp: float = 0.15
    # two-way attenuation per metre of cover. ASSUMPTION. 10 ~ thin/light barrier (lab-like regime);
    # >=30 makes a 24 GHz radar effectively blind to a buried person (see attenuation sweep).
    rubble_atten_db_per_m: float = 10.0
    # ---- vibration (ADXL345-like) ----
    vib_fs_hz: float = 400.0
    vib_noise_g: float = 0.004
    dc_offset_g: float = 1.0
    adc_lsb_g: float = 0.0039
    vib_atten_db_per_m: float = 2.0
    vib_rubble_atten_db_per_m: float = 10.0
    wave_speed_m_s: float = 300.0
    knock_g: float = 0.04
    knock_interval_s: float = 20.0
    footstep_g: float = 0.01
    synthetic: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ScenarioConfig":
        d = dict(d)
        d["persons"] = [PersonSpec(**p) for p in d.get("persons", [])]
        d["machines"] = [MachineSpec(**m) for m in d.get("machines", [])]
        d["vib_sensors"] = [SensorSpec(**s) for s in d.get("vib_sensors", [])]
        return ScenarioConfig(**d)


SCENARIO_NAMES = [
    "no_people", "one_worker", "multi_workers", "survivor_alone", "worker_survivor",
    "multi_workers_survivor", "worker_machinery", "survivor_machinery",
    "worker_survivor_machinery", "multi_workers_survivor_machinery",
]


def _worker(pid: int, rng: np.random.Generator, moving: bool) -> PersonSpec:
    x, y = rng.uniform(-2.5, 2.5), rng.uniform(1.5, 5.0)
    if moving:
        ang, sp = rng.uniform(0, 2 * math.pi), rng.uniform(0.3, 0.8)
        vx, vy, ml = sp * math.cos(ang), sp * math.sin(ang), rng.uniform(0.5, 1.0)
    else:
        vx = vy = 0.0
        ml = rng.uniform(0.03, 0.15)
    return PersonSpec(pid, "worker", x, y, vx, vy, resp_hz=rng.uniform(0.2, 0.35),
                      resp_amp_mm=rng.uniform(3, 6), motion_level=ml)


def _survivor(pid: int, rng: np.random.Generator) -> PersonSpec:
    return PersonSpec(pid, "survivor", rng.uniform(-1.0, 1.0), rng.uniform(2.5, 4.0),
                      buried=True, depth_m=rng.uniform(0.3, 1.0),
                      resp_hz=rng.uniform(0.18, 0.32), resp_amp_mm=rng.uniform(3, 5),
                      motion_level=0.02, knocks=True)


def make_scenario(name: str, seed: int = 0, **overrides: Any) -> ScenarioConfig:
    """Build one of the 10 benchmark scenarios. `overrides` replace top-level config fields."""
    if name not in SCENARIO_NAMES:
        raise ValueError(f"unknown scenario {name!r}; choose from {SCENARIO_NAMES}")
    rng = np.random.default_rng(np.random.SeedSequence([seed, SCENARIO_NAMES.index(name)]))
    persons: List[PersonSpec] = []
    machines: List[MachineSpec] = []
    n_workers = {"no_people": 0, "one_worker": 1, "multi_workers": 3, "survivor_alone": 0,
                 "worker_survivor": 1, "multi_workers_survivor": 3, "worker_machinery": 1,
                 "survivor_machinery": 0, "worker_survivor_machinery": 1,
                 "multi_workers_survivor_machinery": 3}[name]
    for i in range(n_workers):
        persons.append(_worker(len(persons), rng, moving=(i == 0) and (n_workers > 1 or rng.random() < 0.5)))
    if "survivor" in name:
        persons.append(_survivor(len(persons), rng))
    if "machinery" in name:
        machines.append(MachineSpec(-3.0, 2.2, f0_hz=float(rng.uniform(20, 60))))
    cfg = ScenarioConfig(name=name, seed=seed, persons=persons, machines=machines)
    return replace(cfg, **overrides) if overrides else cfg
