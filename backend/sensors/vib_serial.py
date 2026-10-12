"""ESP32 -> laptop vibration framing protocol (defined by this project).

Frame (little-endian):
  A5 5A | ver:u8 | seq:u16 | t_us:u32 | fs_hz:u16 | n_ch:u8 | n_samp:u8
        | payload: int16[n_samp * n_ch] (sample-major, channels interleaved)
        | crc16:u16   (CRC-16/CCITT-FALSE over bytes from `ver` to end of payload)

t_us is the ESP32 micros() of the FIRST sample; it wraps every ~71.6 min and is
unwrapped here. It is DEVICE time. Mapping it onto the host clock is the job of
the sync layer (later phase); until then t0 is device time.
"""
from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from backend.core.types import VibrationPacket
from backend.sensors.base import VibrationSource

log = logging.getLogger(__name__)

HEADER = b"\xA5\x5A"
VERSION = 1
_FIXED = struct.Struct("<BHIHBB")  # ver, seq, t_us, fs, n_ch, n_samp
MAX_CH = 16


def crc16_ccitt(data: bytes, init: int = 0xFFFF) -> int:
    crc = init
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def encode_frame(seq: int, t_us: int, fs_hz: int, samples: np.ndarray) -> bytes:
    """Reference encoder (used by tests and by the simulator's serial mock)."""
    samples = np.asarray(samples, dtype="<i2")
    n_samp, n_ch = samples.shape
    body = _FIXED.pack(VERSION, seq & 0xFFFF, t_us & 0xFFFFFFFF, fs_hz, n_ch, n_samp) + samples.tobytes()
    return HEADER + body + struct.pack("<H", crc16_ccitt(body))


@dataclass
class VibStats:
    frames: int = 0
    crc_errors: int = 0
    dropped_frames: int = 0


class VibFrameParser:
    def __init__(self, scale: float = 1.0) -> None:
        self._buf = bytearray()
        self._scale = scale
        self._last_raw_t: Optional[int] = None
        self._wraps = 0
        self._last_seq: Optional[int] = None
        self.stats = VibStats()

    def feed(self, data: bytes) -> List[VibrationPacket]:
        self._buf.extend(data)
        out: List[VibrationPacket] = []
        while True:
            i = self._buf.find(HEADER)
            if i < 0:
                if len(self._buf) > 1:
                    del self._buf[:-1]
                break
            if i > 0:
                del self._buf[:i]
            if len(self._buf) < 2 + _FIXED.size:
                break
            ver, seq, t_us, fs, n_ch, n_samp = _FIXED.unpack_from(self._buf, 2)
            if ver != VERSION or not (1 <= n_ch <= MAX_CH) or n_samp == 0 or fs == 0:
                self.stats.crc_errors += 1
                del self._buf[:1]
                continue
            total = 2 + _FIXED.size + 2 * n_ch * n_samp + 2
            if len(self._buf) < total:
                break
            body = bytes(self._buf[2:total - 2])
            (crc,) = struct.unpack_from("<H", self._buf, total - 2)
            if crc16_ccitt(body) != crc:
                self.stats.crc_errors += 1
                del self._buf[:1]
                continue
            payload = np.frombuffer(body, dtype="<i2", offset=_FIXED.size).reshape(n_samp, n_ch)
            out.append(VibrationPacket(t0=self._unwrap(t_us), fs=float(fs),
                                       data=payload.astype(np.float64) * self._scale,
                                       seq=seq, synthetic=False))
            self._track_seq(seq)
            self.stats.frames += 1
            del self._buf[:total]
        return out

    def _unwrap(self, t_us: int) -> float:
        if self._last_raw_t is not None and t_us < self._last_raw_t - 2**31:
            self._wraps += 1
        self._last_raw_t = t_us
        return (t_us + self._wraps * 2**32) / 1e6

    def _track_seq(self, seq: int) -> None:
        if self._last_seq is not None:
            gap = (seq - (self._last_seq + 1)) & 0xFFFF
            if gap:
                self.stats.dropped_frames += gap
                log.warning("vibration: %d frame(s) dropped (seq gap)", gap)
        self._last_seq = seq


class SerialVibrationSource(VibrationSource):
    def __init__(self, port: str, baud: int = 921600, scale: float = 1.0) -> None:
        try:
            import serial
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("pyserial not installed: pip install pyserial") from e
        self._ser = serial.Serial(port, baud, timeout=0)
        self._parser = VibFrameParser(scale)
        log.info("Vibration source opened on %s @ %d", port, baud)

    @property
    def stats(self) -> VibStats:
        return self._parser.stats

    def read(self) -> List[VibrationPacket]:
        data = self._ser.read(self._ser.in_waiting or 1)
        return self._parser.feed(data) if data else []

    def close(self) -> None:
        self._ser.close()
