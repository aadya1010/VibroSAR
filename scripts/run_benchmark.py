"""Build synthetic dataset -> train A/B detectors -> evaluate (10 scenarios) -> write reports/benchmark_report.md.

All numbers are on SYNTHETIC data and say nothing about real-world performance.
Usage: python scripts/run_benchmark.py [--rebuild] [--seeds 16]
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.core.types import EvidenceState  # noqa: E402
from backend.fusion.state_machine import StateConfig, StateMachine  # noqa: E402
from backend.models.baseline import FEATURE_SETS, Detector, metrics  # noqa: E402
from backend.models.features import EXCL_COLS, TRK_COLS, VIB_COLS  # noqa: E402
from backend.models.dataset import build_dataset  # noqa: E402
from simulator.scenario import SCENARIO_NAMES  # noqa: E402

DATA = Path("data/synthetic/windows_v1.csv")


def md(df: pd.DataFrame, floatfmt: int = 2) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(f"{v:.{floatfmt}f}" if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(out)


def recording_states(df: pd.DataFrame, p: np.ndarray, thr: float) -> pd.DataFrame:
    # state thresholds are tied to the detector's validated operating point (FPR-constrained threshold)
    cfg = StateConfig(p_uncertain=thr / 2, p_possible=thr, p_high=thr + 0.5 * (1 - thr))
    rows = []
    d = df.assign(p=p)
    for (sc, seed), g in d.groupby(["scenario", "seed"]):
        sm, first_possible, final, reached_high = StateMachine(cfg), None, None, False
        for _, w in g.sort_values("t0").iterrows():
            st = sm.update(w.p, quality=min(1.0, w.pilot_recovery), interference=w.interference)
            if first_possible is None and st in (EvidenceState.POSSIBLE_SURVIVOR, EvidenceState.HIGH_CONFIDENCE_CANDIDATE):
                first_possible = w.t1
            reached_high |= st == EvidenceState.HIGH_CONFIDENCE_CANDIDATE
            final = st
        rows.append(dict(scenario=sc, seed=seed, y=int(g.y_survivor.iloc[0]), reached_possible=first_possible is not None,
                         reached_high=reached_high, t_first_possible=first_possible if first_possible is not None else np.nan))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--seeds", type=int, default=16)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if a.rebuild or not DATA.exists():
        df = build_dataset(range(a.seeds), workers=min(4, os.cpu_count() or 1))
        DATA.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA, index=False)
    df = pd.read_csv(DATA)
    tr, va, te = df[df.seed < 10], df[df.seed.isin([10, 11])], df[df.seed >= 12]
    print(f"rows train={len(tr)} val={len(va)} test={len(te)}  (SYNTHETIC; split by seed)")

    report = ["# VibroSAR benchmark (SYNTHETIC data only)\n",
              "Train seeds 0-9, calibration/threshold seeds 10-11, test seeds 12-15. Threshold = max recall with FPR <= 5% on validation.\n"]
    summary, probs = {}, {}
    for name, cols in FEATURE_SETS.items():
        det = Detector(cols).fit(tr, va)
        det.save(f"models/checkpoints/{name}.joblib")
        p = det.predict_proba(te)
        probs[name] = (det, p)
        summary[name] = metrics(te.y_survivor.to_numpy(), p, det.thr) | {"thr": det.thr}
    naive = te.naive_flag.to_numpy()
    nm = metrics(te.y_survivor.to_numpy(), naive.astype(float), 0.5)
    summary["naive_breathing_anywhere"] = nm
    S = pd.DataFrame(summary).T[["precision", "recall", "f1", "fpr", "fnr", "auroc", "ece", "n"]]
    S.index.name = "detector"
    report += ["## Overall (test windows)\n", md(S.reset_index(), 3), "\n"]
    print(S.round(3))

    rows = []
    for sc in SCENARIO_NAMES:
        m = te.scenario == sc
        r = dict(scenario=sc, truth="survivor" if te[m].y_survivor.iloc[0] else "none",
                 naive=te[m].naive_flag.mean())
        for name, (det, p) in probs.items():
            r[name] = float((p[m.to_numpy()] >= det.thr).mean())
        rows.append(r)
    P = pd.DataFrame(rows)
    report += ["## Fraction of windows flagged positive, per scenario (want ~0 for 'none', ~1 for 'survivor')\n", md(P, 2), "\n"]
    print(P.round(2).to_string(index=False))

    for name, (det, p) in probs.items():
        R = recording_states(te.reset_index(drop=True), p, det.thr)
        agg = R.groupby("scenario").agg(truth=("y", "first"), reached_possible=("reached_possible", "mean"),
                                        reached_high=("reached_high", "mean"), t_first_possible_s=("t_first_possible", "median")).reset_index()
        report += [f"## Recording-level states, detector {name}\n", md(agg, 2), "\n"]
        print(f"\n{name}: recordings reaching POSSIBLE: positives={R[R.y==1].reached_possible.mean():.2f} negatives(false alarm)={R[R.y==0].reached_possible.mean():.2f}")
        if name == "A_full_sim":
            hit = (te.best_range_m > 0) & (te.y_survivor == 1) & (p >= det.thr)
            err = (te.best_range_m - te.truth_range_m).abs()[hit]
            report.append(f"Localization (A, true-positive windows with a radar candidate): n={int(hit.sum())}, median error {err.median():.2f} m, p90 {err.quantile(.9):.2f} m (range bin = 0.1 m)\n")
    # ---- ablation: which evidence actually carries the performance? ----
    abl = {"B full (vib + trk + camera)": EXCL_COLS + TRK_COLS + VIB_COLS, "B minus radar tracks": EXCL_COLS + VIB_COLS,
           "vibration only": VIB_COLS,
           "vibration, no interference/pilot feats": [c for c in VIB_COLS if c not in ("pilot_recovery", "interference", "ref_rms_g", "coh_raw_ref")]}
    rows = []
    for n, cols in abl.items():
        d = Detector(cols).fit(tr, va)
        m = metrics(te.y_survivor.to_numpy(), d.predict_proba(te), d.thr)
        rows.append(dict(model=n, recall=m["recall"], fpr=m["fpr"], auroc=m["auroc"]))
    AB = pd.DataFrame(rows)
    report += ["## Ablation (hardware-realistic evidence). Radar-track features carry much of B's score ONLY because the simulator lets a "
               "24 GHz tracker partly see through light cover (10 dB/m). Treat vibration-only as the conservative estimate.\n", md(AB, 3), "\n"]
    print("\nABLATION"); print(AB.round(3).to_string(index=False))

    # ---- attenuation sweep: models trained at 10 dB/m, tested at other cover thickness ----
    rows = []
    for att in (5, 10, 20, 30, 40):
        ds = build_dataset([12, 13], overrides=dict(rubble_atten_db_per_m=float(att)), workers=min(4, os.cpu_count() or 1))
        r = dict(atten_db_per_m=att)
        for name, (det, _) in probs.items():
            pp = det.predict_proba(ds) >= det.thr
            r[f"{name} TPR"] = float(pp[ds.y_survivor.to_numpy() == 1].mean())
            r[f"{name} FPR"] = float(pp[ds.y_survivor.to_numpy() == 0].mean())
        rows.append(r)
    SW = pd.DataFrame(rows)
    report += ["## Cover-attenuation sweep (train at 10 dB/m, test at other values; 24 GHz radar assumption)\n", md(SW, 2), "\n"]
    print("\nATTENUATION SWEEP"); print(SW.round(2).to_string(index=False))

    Path("reports").mkdir(exist_ok=True)
    Path("reports/benchmark_report.md").write_text("\n".join(report))
    json.dump({k: {a: float(b) for a, b in v.items()} for k, v in summary.items()}, open("reports/metrics.json", "w"), indent=2)
    print("\nwrote reports/benchmark_report.md")


if __name__ == "__main__":
    main()
