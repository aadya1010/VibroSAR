"""Synthetic dataset builder: scenarios x seeds -> window-feature table with labels.

All rows are SYNTHETIC (column `synthetic`=1). Splits are by SEED (never by window) to avoid leakage.
"""
from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from backend.models.features import FeatureConfig, features_from_sim
from simulator.scenario import SCENARIO_NAMES, make_scenario
from simulator.simulate import SimResult, run_scenario

log = logging.getLogger(__name__)


def label_windows(df: pd.DataFrame, r: SimResult) -> pd.DataFrame:
    surv = [i for i, b in enumerate(r.buried) if b]
    truth_range = float(np.hypot(r.cfg.persons[surv[0]].x_m, r.cfg.persons[surv[0]].y_m)) if surv else -1.0
    kt = np.array([t for _, t in r.knock_times])
    out = df.copy()
    out["y_survivor"] = int(bool(surv))
    out["truth_range_m"] = truth_range
    out["scenario"] = r.cfg.name
    out["seed"] = r.cfg.seed
    out["n_visible"] = int(sum(1 for b in r.buried if not b))
    out["machine_on"] = [float(r.machine_active[int(a * r.vib_fs):int(b * r.vib_fs)].mean()) for a, b in zip(df.t0, df.t1)]
    out["knock_in_win"] = [int(((kt >= a) & (kt < b)).any()) if len(kt) else 0 for a, b in zip(df.t0, df.t1)]
    out["synthetic"] = 1
    return out


def _one(args):
    name, seed, duration, overrides, fcfg = args
    r = run_scenario(make_scenario(name, seed, duration_s=duration, **overrides))
    return label_windows(features_from_sim(r, fcfg), r)


def build_dataset(seeds: Iterable[int], names: Optional[Iterable[str]] = None, duration_s: float = 120.0,
                  overrides: Optional[dict] = None, fcfg: FeatureConfig = FeatureConfig(),
                  workers: int = 4) -> pd.DataFrame:
    jobs = [(n, s, duration_s, overrides or {}, fcfg) for s in seeds for n in (names or SCENARIO_NAMES)]
    log.info("building %d recordings (SYNTHETIC)", len(jobs))
    if workers <= 1:
        parts = [_one(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            parts = list(ex.map(_one, jobs))
    return pd.concat(parts, ignore_index=True)
