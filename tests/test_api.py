import pytest
from fastapi.testclient import TestClient

from backend.api.app import CKPT, app

pytestmark = pytest.mark.skipif(not CKPT["B_hardware_like"].exists(), reason="run scripts/run_benchmark.py first")


def test_health_and_replay_ws():
    c = TestClient(app)
    assert c.get("/api/health").json()["synthetic_replay"] is True
    with c.websocket_connect("/ws/replay") as ws:
        ws.send_json(dict(scenario="survivor_alone", seed=3, model="B_hardware_like", speed=1000))
        msgs = []
        while True:
            m = ws.receive_json()
            msgs.append(m)
            if m.get("done") or m.get("error"):
                break
    frames = [m["frame"] for m in msgs if "frame" in m]
    assert len(frames) >= 8 and all(0 <= f["p"] <= 1 for f in frames)
    assert frames[-1]["state"] in ("NO_EVIDENCE", "UNCERTAIN", "POSSIBLE_SURVIVOR", "HIGH_CONFIDENCE_CANDIDATE")
