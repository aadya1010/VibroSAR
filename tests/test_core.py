import numpy as np
import pytest

from backend.core.clock import RingBuffer
from backend.core.config import load_config
from backend.core.types import (Candidate, EvidenceState, RadarCaps, RadarMode,
                                VibrationPacket)


def test_vibration_packet_validation():
    with pytest.raises(ValueError):
        VibrationPacket(0.0, 500.0, np.zeros(10))
    p = VibrationPacket(1.0, 500.0, np.zeros((50, 2)))
    assert p.n_samples == 50 and p.n_channels == 2 and p.t_end == pytest.approx(1.1)


def test_candidate_confidence_range():
    with pytest.raises(ValueError):
        Candidate(1, 0, 0, EvidenceState.UNCERTAIN, 1.5, 0.5, 0.5)
    c = Candidate(1, None, None, EvidenceState.NO_EVIDENCE, 0.0, 0.5, 0.2)
    assert c.depth_m is None  # depth must stay None unless calibrated


def test_radar_caps_iq_requires_phase():
    with pytest.raises(ValueError):
        RadarCaps(RadarMode.IQ, False, 10, 1)


def test_ring_buffer_wrap():
    rb = RingBuffer(5, 1)
    rb.append(np.arange(3).reshape(-1, 1))
    rb.append(np.arange(3, 7).reshape(-1, 1))
    assert len(rb) == 5
    assert rb.latest(5).ravel().tolist() == [2, 3, 4, 5, 6]
    assert rb.latest(2).ravel().tolist() == [5, 6]


def test_default_config_loads():
    cfg = load_config("configs/default.yaml")
    assert cfg.radar.mode == RadarMode.TRACKS and cfg.vibration.fs_hz == 400
