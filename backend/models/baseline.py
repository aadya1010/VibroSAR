"""Classical-ML baseline detector: HistGradientBoosting + isotonic calibration + FPR-constrained threshold.

Input  : window feature row (see features.py), ~35 columns.
Output : calibrated score in [0, 1] for 'buried survivor evidence present'. It is a MODEL SCORE
         calibrated on SYNTHETIC validation data - NOT a validated probability of a real survivor.
Target : y_survivor (1 if the scenario contains a buried person).
Loss   : log-loss (gradient boosting). Calibration: isotonic regression on a held-out validation split.
Why    : few thousand rows, tabular engineered features, need explainability + CPU speed -> trees, not deep nets.
Failure: learns simulator artefacts; collapses when real noise differs (sim-to-real gap).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from backend.models.features import EXCL_COLS, RESP_COLS, TRK_COLS, VIB_COLS

FEATURE_SETS: Dict[str, List[str]] = {
    "A_full_sim": RESP_COLS + EXCL_COLS + TRK_COLS + VIB_COLS,          # needs raw-phase radar
    "B_hardware_like": EXCL_COLS + TRK_COLS + VIB_COLS,                 # RD-03D tracks + camera + ADXL345
}


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def metrics(y: np.ndarray, p: np.ndarray, thr: float) -> Dict[str, float]:
    pred = p >= thr
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return dict(precision=prec, recall=rec, f1=2 * prec * rec / max(prec + rec, 1e-12), fpr=fp / max(fp + tn, 1),
                fnr=fn / max(fn + tp, 1), auroc=float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
                ece=ece(y, p), n=len(y))


@dataclass
class Detector:
    cols: List[str]
    seed: int = 0
    thr: float = 0.5

    def __post_init__(self) -> None:
        self.clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4,
                                                  l2_regularization=1.0, random_state=self.seed)
        self.iso = None

    def fit(self, train: pd.DataFrame, val: pd.DataFrame, max_fpr: float = 0.05) -> "Detector":
        self.clf.fit(train[self.cols], train.y_survivor)
        raw = self.clf.predict_proba(val[self.cols])[:, 1]
        self.iso = IsotonicRegression(out_of_bounds="clip").fit(raw, val.y_survivor)
        p, y = self.iso.predict(raw), val.y_survivor.to_numpy()
        best = None
        for t in np.unique(np.round(p, 4)):               # smallest-FPR-compliant threshold with max recall
            m = metrics(y, p, t)
            if m["fpr"] <= max_fpr and (best is None or m["recall"] > best[1]):
                best = (t, m["recall"])
        self.thr = float(best[0]) if best else 0.5
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.iso.predict(self.clf.predict_proba(df[self.cols])[:, 1])

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(dict(cols=self.cols, clf=self.clf, iso=self.iso, thr=self.thr, synthetic_trained=True), path)

    @staticmethod
    def load(path: str | Path) -> "Detector":
        d = joblib.load(path)
        det = Detector(d["cols"])
        det.clf, det.iso, det.thr = d["clf"], d["iso"], d["thr"]
        return det
