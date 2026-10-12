"""YAML configuration with validation (fail early, with clear messages)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml

from backend.core.types import RadarMode


@dataclass(frozen=True)
class RadarConfig:
    mode: RadarMode
    port: str
    baud: int
    max_targets: int


@dataclass(frozen=True)
class VibrationConfig:
    port: str
    baud: int
    fs_hz: float
    channel_names: List[str]
    scale: float  # counts -> physical units; set per sensor after calibration


@dataclass(frozen=True)
class AppConfig:
    radar: RadarConfig
    vibration: VibrationConfig
    log_level: str


def _need(d: dict, key: str, where: str):
    if key not in d:
        raise ValueError(f"config: missing '{key}' in {where}")
    return d[key]


def load_config(path: str | Path) -> AppConfig:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("config: top level must be a mapping")
    r = _need(raw, "radar", "root")
    v = _need(raw, "vibration", "root")
    try:
        mode = RadarMode(_need(r, "mode", "radar"))
    except ValueError as e:
        raise ValueError(f"config: invalid radar.mode ({e})") from e
    radar = RadarConfig(mode, str(_need(r, "port", "radar")),
                        int(_need(r, "baud", "radar")), int(r.get("max_targets", 3)))
    fs = float(_need(v, "fs_hz", "vibration"))
    if fs <= 0:
        raise ValueError("config: vibration.fs_hz must be > 0")
    names = list(_need(v, "channel_names", "vibration"))
    if not names:
        raise ValueError("config: vibration.channel_names must not be empty")
    vib = VibrationConfig(str(_need(v, "port", "vibration")), int(_need(v, "baud", "vibration")),
                          fs, names, float(v.get("scale", 1.0)))
    return AppConfig(radar, vib, str(raw.get("logging", {}).get("level", "INFO")))
