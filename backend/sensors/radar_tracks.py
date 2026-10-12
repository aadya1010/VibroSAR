"""RD-03D (Mode B, processed tracks) parser and serial source.

ASSUMPTION TO VERIFY: RD-03D reports use an LD2450-style 30-byte frame:
    AA FF 03 00 | 3 x [x:u16 y:u16 speed:u16 res:u16] | 55 CC
with sign-magnitude values where bit15 set => positive. Confirm against your
own serial capture (tools/capture_rd03d.py) before trusting the numbers.

Mode B gives NO phase / respiration waveform. capabilities().has_phase is False.
"""
from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from typing import List, Optional

from backend.core.clock import Clock
from backend.core.types import RadarCaps, RadarMode, RadarPacket, RadarTrack
from backend.sensors.base import RadarSource

log = logging.getLogger(__name__)

HEADER = b"\xAA\xFF\x03\x00"
TAIL = b"\x55\xCC"
FRAME_LEN = 30
_TARGET = struct.Struct("<HHHH")


def _sign_mag(raw: int) -> int:
    mag = raw & 0x7FFF
    return mag if raw & 0x8000 else -mag


@dataclass
class ParserStats:
    frames: int = 0
    bad_frames: int = 0


class Rd03dParser:
    def __init__(self) -> None:
        self._buf = bytearray()
        self.stats = ParserStats()

    def feed(self, data: bytes, t: float) -> List[RadarPacket]:
        self._buf.extend(data)
        out: List[RadarPacket] = []
        while True:
            i = self._buf.find(HEADER)
            if i < 0:
                keep = len(HEADER) - 1  # a partial header may straddle chunks
                if len(self._buf) > keep:
                    del self._buf[:-keep]
                break
            if i > 0:
                del self._buf[:i]
            if len(self._buf) < FRAME_LEN:
                break
            if bytes(self._buf[FRAME_LEN - 2:FRAME_LEN]) != TAIL:
                self.stats.bad_frames += 1
                del self._buf[:1]  # resync
                continue
            out.append(self._decode(bytes(self._buf[:FRAME_LEN]), t))
            self.stats.frames += 1
            del self._buf[:FRAME_LEN]
        return out

    @staticmethod
    def _decode(frame: bytes, t: float) -> RadarPacket:
        tracks = []
        for k in range(3):
            raw = _TARGET.unpack_from(frame, 4 + 8 * k)
            if raw == (0, 0, 0, 0):
                continue  # empty slot
            x, y, v, res = raw
            tracks.append(RadarTrack(_sign_mag(x), _sign_mag(y), _sign_mag(v), res))
        return RadarPacket(t=t, tracks=tuple(tracks), synthetic=False)


class SerialRadarSource(RadarSource):
    def __init__(self, port: str, baud: int = 256000, clock: Optional[Clock] = None) -> None:
        try:
            import serial  # pyserial
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("pyserial not installed: pip install pyserial") from e
        self._ser = serial.Serial(port, baud, timeout=0)
        self._clock = clock or Clock()
        self._parser = Rd03dParser()
        log.info("RD-03D opened on %s @ %d", port, baud)

    def capabilities(self) -> RadarCaps:
        # Frame rate ~10 Hz is typical for this class of module; verify from capture.
        return RadarCaps(RadarMode.TRACKS, has_phase=False, frame_rate_hz=10.0, max_targets=3)

    def read(self) -> List[RadarPacket]:
        data = self._ser.read(self._ser.in_waiting or 1)
        return self._parser.feed(data, self._clock.now()) if data else []

    def close(self) -> None:
        self._ser.close()
