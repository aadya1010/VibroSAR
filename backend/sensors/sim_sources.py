"""Simulator-backed sources implementing the same interfaces as real hardware."""
from __future__ import annotations

from typing import List

from backend.core.types import RadarCaps, RadarMode, RadarPacket, VibrationPacket
from backend.sensors.base import RadarSource, VibrationSource


class SimRadarSource(RadarSource):
    """Replays a SimResult as a stream. `mode` picks Mode A (IQ) or Mode B (TRACKS)."""

    def __init__(self, result, mode: str = "TRACKS", chunk_s: float = 1.0) -> None:
        self._r, self._mode, self._chunk = result, mode, chunk_s
        self._packets = result.radar_packets(mode)
        self._i = 0

    def capabilities(self) -> RadarCaps:
        c = self._r.cfg
        if self._mode == "IQ":
            return RadarCaps(RadarMode.IQ, True, c.iq_fs_hz, 1, synthetic=True)
        return RadarCaps(RadarMode.TRACKS, False, c.track_fs_hz, 3, synthetic=True)

    @property
    def finished(self) -> bool:
        return self._i >= len(self._packets)

    def read(self) -> List[RadarPacket]:
        if self.finished:
            return []
        t_end = self._packets[self._i].t + self._chunk
        j = self._i
        while j < len(self._packets) and self._packets[j].t < t_end:
            j += 1
        out, self._i = self._packets[self._i:j], j
        return out

    def close(self) -> None:
        pass


class SimVibrationSource(VibrationSource):
    def __init__(self, result, frame_samples: int = 20, frames_per_read: int = 20) -> None:
        self._packets = result.vibration_packets(frame_samples)
        self._n, self._i = frames_per_read, 0

    @property
    def finished(self) -> bool:
        return self._i >= len(self._packets)

    def read(self) -> List[VibrationPacket]:
        out = self._packets[self._i:self._i + self._n]
        self._i += len(out)
        return out

    def close(self) -> None:
        pass
