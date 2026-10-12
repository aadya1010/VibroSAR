"""Evidence state machine: probabilities -> NO_EVIDENCE / UNCERTAIN / POSSIBLE_SURVIVOR / HIGH_CONFIDENCE_CANDIDATE.

Rules (all configurable): upgrades need persistence (n consecutive windows over a threshold);
high interference or an unsafe canceller caps the state; poor signal quality caps it at UNCERTAIN;
downgrades happen after `n_down` windows below the lower hysteresis level. Unknown beats false certainty.
"""
from __future__ import annotations

from dataclasses import dataclass

from backend.core.types import EvidenceState

_ORDER = [EvidenceState.NO_EVIDENCE, EvidenceState.UNCERTAIN, EvidenceState.POSSIBLE_SURVIVOR,
          EvidenceState.HIGH_CONFIDENCE_CANDIDATE]


@dataclass(frozen=True)
class StateConfig:
    p_uncertain: float = 0.25
    p_possible: float = 0.5
    p_high: float = 0.85
    n_possible: int = 2
    n_high: int = 4
    n_down: int = 2
    hysteresis: float = 0.7           # drop level = threshold * hysteresis
    interference_cap: float = 0.9     # above this: max POSSIBLE
    quality_min: float = 0.5          # below this: max UNCERTAIN


class StateMachine:
    def __init__(self, cfg: StateConfig = StateConfig()) -> None:
        self.cfg = cfg
        self.state = EvidenceState.NO_EVIDENCE
        self._up_p = self._up_h = self._down = 0

    def update(self, p: float, quality: float = 1.0, interference: float = 0.0) -> EvidenceState:
        c = self.cfg
        self._up_p = self._up_p + 1 if p >= c.p_possible else 0
        self._up_h = self._up_h + 1 if p >= c.p_high else 0
        target = EvidenceState.NO_EVIDENCE
        if p >= c.p_uncertain:
            target = EvidenceState.UNCERTAIN
        if self._up_p >= c.n_possible:
            target = EvidenceState.POSSIBLE_SURVIVOR
        if self._up_h >= c.n_high:
            target = EvidenceState.HIGH_CONFIDENCE_CANDIDATE
        cap = EvidenceState.HIGH_CONFIDENCE_CANDIDATE
        if interference > c.interference_cap:
            cap = EvidenceState.POSSIBLE_SURVIVOR
        if quality < c.quality_min:
            cap = EvidenceState.UNCERTAIN
        target = min(target, cap, key=_ORDER.index)
        cur, tgt = _ORDER.index(self.state), _ORDER.index(target)
        if tgt > cur:
            self.state, self._down = target, 0
        elif tgt < cur:
            self._down += 1
            lower_ok = p < c.hysteresis * {1: c.p_uncertain, 2: c.p_possible, 3: c.p_high}[cur] or quality < c.quality_min
            if self._down >= c.n_down and lower_ok or _ORDER.index(cap) < cur:
                self.state, self._down = target, 0
        else:
            self._down = 0
        return self.state
