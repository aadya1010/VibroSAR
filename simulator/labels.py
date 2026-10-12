"""Per-window ground-truth labels (for training and evaluation)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from simulator.simulate import SimResult


def window_labels(r: SimResult, win_s: float = 20.0, hop_s: float = 5.0) -> pd.DataFrame:
    rows, T = [], r.cfg.duration_s
    surv = [i for i, b in enumerate(r.buried) if b]
    knock_t = np.array([t for _, t in r.knock_times]) if r.knock_times else np.array([])
    tgt = 0  # sensor 0 is the "target" sensor by convention
    t0 = 0.0
    while t0 + win_s <= T + 1e-9:
        t1 = t0 + win_s
        a, b = int(t0 * r.vib_fs), int(t1 * r.vib_fs)
        ia, ib = np.searchsorted(r.iq_t, [t0, t1])
        resp_hz = r.cfg.persons[surv[0]].resp_hz if surv else np.nan
        rows.append(dict(
            t0=t0, t1=t1,
            survivor_buried=bool(surv),
            n_visible_people=int(sum(1 for bb in r.buried if not bb)),
            machine_active_frac=float(r.machine_active[a:b].mean()),
            machine_rms_target_g=float(np.sqrt(np.mean(r.vib_components["machine"][a:b, tgt] ** 2))),
            knock_in_window=bool(((knock_t >= t0) & (knock_t < t1)).any()) if len(knock_t) else False,
            survivor_resp_hz=resp_hz))
        t0 += hop_s
    return pd.DataFrame(rows)
