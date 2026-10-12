"""Core data types shared by every VibroSAR module.

Design rule: downstream code (DSP, fusion, UI) only ever sees these types,
never raw sensor bytes. Every packet carries a `synthetic` flag so simulated
data can never be mistaken for real measurements.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import numpy as np


class RadarMode(str, Enum):
    IQ = "IQ"          # Mode A: raw I/Q / phase available
    TRACKS = "TRACKS"  # Mode B: processed targets only (e.g. RD-03D)
    SIM = "SIM"        # Mode C: simulator


class EvidenceState(str, Enum):
    NO_EVIDENCE = "NO_EVIDENCE"
    UNCERTAIN = "UNCERTAIN"
    POSSIBLE_SURVIVOR = "POSSIBLE_SURVIVOR"
    HIGH_CONFIDENCE_CANDIDATE = "HIGH_CONFIDENCE_CANDIDATE"


@dataclass(frozen=True)
class RadarCaps:
    """What a radar source can actually provide. Downstream code must check this."""
    mode: RadarMode
    has_phase: bool
    frame_rate_hz: float
    max_targets: int
    synthetic: bool = False

    def __post_init__(self) -> None:
        if self.frame_rate_hz <= 0:
            raise ValueError("frame_rate_hz must be > 0")
        if self.mode == RadarMode.IQ and not self.has_phase:
            raise ValueError("IQ mode implies has_phase=True")


@dataclass(frozen=True)
class RadarTrack:
    """One processed radar target (Mode B). Coordinates in sensor frame."""
    x_mm: int
    y_mm: int
    speed_cm_s: int
    resolution_mm: int

    @property
    def range_mm(self) -> float:
        return math.hypot(self.x_mm, self.y_mm)


@dataclass
class RadarPacket:
    t: float                                   # seconds, host monotonic clock
    tracks: Tuple[RadarTrack, ...] = ()
    iq: Optional[np.ndarray] = None            # Mode A only
    synthetic: bool = False

    def __post_init__(self) -> None:
        if self.iq is not None and self.iq.ndim not in (1, 2):
            raise ValueError("iq must be 1-D or 2-D")


@dataclass
class VibrationPacket:
    t0: float          # time of first sample (device or host clock, see source docs)
    fs: float          # sampling rate, Hz
    data: np.ndarray   # shape (n_samples, n_channels), float
    seq: int = 0
    synthetic: bool = False

    def __post_init__(self) -> None:
        if self.fs <= 0:
            raise ValueError("fs must be > 0")
        if self.data.ndim != 2:
            raise ValueError("data must have shape (n_samples, n_channels)")

    @property
    def n_samples(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.data.shape[1])

    @property
    def t_end(self) -> float:
        return self.t0 + self.n_samples / self.fs


@dataclass(frozen=True)
class PersonDetection:
    """A visible person from the camera pipeline (feeds the exclusion map)."""
    t: float
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


@dataclass(frozen=True)
class Evidence:
    name: str
    value: float
    unit: str = ""
    quality: float = 1.0   # 0..1, how much to trust this measurement
    note: str = ""


@dataclass
class Candidate:
    """Final output object. Values must be derived from evidence, never invented."""
    id: int
    x_m: Optional[float]
    y_m: Optional[float]
    state: EvidenceState
    confidence: float                  # model score; NOT a calibrated survival probability unless validated
    signal_quality: float
    interference_level: float
    localization_sigma_m: Optional[float] = None
    depth_m: Optional[float] = None    # stays None unless a calibration experiment supports it
    evidence: list[Evidence] = field(default_factory=list)
    synthetic: bool = False

    def __post_init__(self) -> None:
        for name in ("confidence", "signal_quality", "interference_level"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {v}")
