"""FastAPI dashboard backend. SYNTHETIC replay only (hardware on hold).

Run:  uvicorn backend.api.app:app --port 8000      then open http://localhost:8000
WS  /ws/replay : send {"scenario","seed","model","speed"}; receives one JSON frame per analysis window.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from backend.explain.evidence_builder import build_evidence
from backend.fusion.state_machine import StateConfig, StateMachine
from backend.models.baseline import Detector
from backend.models.features import FeatureConfig, features_from_sim
from backend.tracking.exclusion import exclusion_mask
from simulator.scenario import SCENARIO_NAMES, make_scenario
from simulator.simulate import run_scenario

ROOT = Path(__file__).resolve().parents[2]
CKPT = {"A_full_sim": ROOT / "models/checkpoints/A_full_sim.joblib",
        "B_hardware_like": ROOT / "models/checkpoints/B_hardware_like.joblib"}
log = logging.getLogger(__name__)
app = FastAPI(title="VibroSAR dashboard (SYNTHETIC replay)")


def _segments(mask: np.ndarray, r_axis: np.ndarray, bin_m: float = 0.1):
    segs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if (not m or i == len(mask) - 1) and start is not None:
            end = i if m else i - 1
            segs.append([float(r_axis[start] - bin_m / 2), float(r_axis[end] + bin_m / 2)])
            start = None
    return segs


def build_replay(scenario: str, seed: int, model: str, duration_s: float = 120.0) -> dict:
    if model not in CKPT or not CKPT[model].exists():
        raise ValueError(f"checkpoint for {model} missing: run scripts/run_benchmark.py first")
    r = run_scenario(make_scenario(scenario, seed, duration_s=duration_s))
    df = features_from_sim(r, FeatureConfig())
    det = Detector.load(CKPT[model])
    p = det.predict_proba(df)
    thr = det.thr
    sm = StateMachine(StateConfig(p_uncertain=thr / 2, p_possible=thr, p_high=thr + 0.5 * (1 - thr)))
    surv = [q for q in r.cfg.persons if q.buried]
    frames = []
    for i, row in df.iterrows():
        st = sm.update(float(p[i]), quality=min(1.0, float(row.pilot_recovery)), interference=float(row.interference))
        sel = (r.cam_t >= row.t0) & (r.cam_t <= row.t1)
        xy = r.cam_xy[sel].reshape(-1, 2)
        xy = xy[~np.isnan(xy).any(axis=1)]
        last = r.cam_xy[sel][-1].reshape(-1, 2) if sel.any() else np.empty((0, 2))
        last = last[~np.isnan(last).any(axis=1)]
        ts = r.tracks_arr[(r.tracks_t >= row.t0) & (r.tracks_t <= row.t1)]
        trk = np.unique(ts.reshape(-1, 4)[ts.reshape(-1, 4).any(axis=1)][:, :2] / 1000.0, axis=0)[:12]
        excl = exclusion_mask(r.cam_t, r.cam_xy, row.t0, row.t1, r.range_axis)
        cand = float(row.best_range_m) if model.startswith("A") and row.best_bin >= 0 and p[i] >= thr else None
        frames.append(dict(
            t0=float(row.t0), t1=float(row.t1), p=float(p[i]), thr=float(thr), state=st.value,
            signal_quality=float(min(1.0, row.pilot_recovery)), interference=float(row.interference),
            candidate_range_m=cand, visible_people=last.round(2).tolist(), radar_tracks=trk.round(2).tolist(),
            excluded_ranges=_segments(excl, r.range_axis), evidence=build_evidence(row, model),
            truth=[dict(x=q.x_m, y=q.y_m, depth=q.depth_m) for q in surv]))
    return dict(synthetic=True, scenario=scenario, seed=seed, model=model, frames=frames)


@app.get("/api/health")
def health():
    return dict(ok=True, synthetic_replay=True, scenarios=SCENARIO_NAMES, models={k: v.exists() for k, v in CKPT.items()})


@app.websocket("/ws/replay")
async def replay(ws: WebSocket):
    await ws.accept()
    try:
        req = await ws.receive_json()
        speed = max(0.1, float(req.get("speed", 1.0)))
        try:
            data = await asyncio.to_thread(build_replay, req.get("scenario", "survivor_alone"),
                                           int(req.get("seed", 0)), req.get("model", "B_hardware_like"))
        except Exception as e:  # report clearly to the UI instead of dropping the socket
            await ws.send_json(dict(error=str(e)))
            return
        for f in data["frames"]:
            await ws.send_json(dict(frame=f, scenario=data["scenario"], model=data["model"]))
            await asyncio.sleep(1.0 / speed)
        await ws.send_json(dict(done=True))
    except WebSocketDisconnect:
        log.info("client disconnected")


@app.get("/")
def index():
    return FileResponse(ROOT / "frontend/index.html")
