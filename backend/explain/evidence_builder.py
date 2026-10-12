"""Rule-based evidence text from window features (deterministic; no LLM, nothing is invented)."""
from __future__ import annotations

from typing import Dict, List


def build_evidence(r, model: str) -> List[Dict[str, str]]:
    ev: List[Dict[str, str]] = []

    def add(name, note, strength):
        ev.append(dict(name=name, note=note, strength=strength))

    if r.knock_triplet_score >= 0.66:
        add("Knock pattern", f"repeated impacts ~0.5 s apart (score {r.knock_triplet_score:.2f}, {int(r.n_knock_like)} knock-like events)", "strong")
    elif r.n_knock_like > 0:
        add("Knock-like events", f"{int(r.n_knock_like)} isolated events, no clear pattern", "weak")
    else:
        add("Knock pattern", "none detected", "none")
    if r.n_step_like >= 3:
        add("Footsteps", f"{int(r.n_step_like)} footstep-like impacts (workers, NOT survivor evidence)", "excluded")
    if r.interference > 0.5:
        add("Machinery interference", f"level {r.interference:.2f}, cancellation gain {r.cancel_gain_db:.0f} dB", "warning")
    if r.pilot_recovery < 0.7:
        add("Canceller safety", f"pilot recovery {r.pilot_recovery:.2f} (<0.7: output may have lost signal)", "warning")
    if model.startswith("A"):
        if r.resp_snr_db >= 10 and r.best_bin >= 0:
            add("Breathing-like micro-motion", f"{r.best_range_m:.1f} m, {r.resp_rate_hz * 60:.0f}/min, SNR {r.resp_snr_db:.0f} dB, outside visible-person zones", "strong")
        if r.n_resp_cands_excl > 0:
            add("Visible-person exclusion", f"{int(r.n_resp_cands_excl)} breathing signal(s) inside visible-person zones ignored", "excluded")
        if r.n_resp_ghost > 0:
            add("Ghost rejection", f"{int(r.n_resp_ghost)} duplicate (multipath) breathing signal(s) rejected", "excluded")
    if r.n_cam_hits > 0:
        add("Camera", f"visible people present ({int(r.n_cam_hits)} detections in window)", "info")
    return ev
