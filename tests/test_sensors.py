import struct

import numpy as np

from backend.sensors.radar_tracks import FRAME_LEN, HEADER, TAIL, Rd03dParser
from backend.sensors.vib_serial import VibFrameParser, encode_frame


def _enc(v: int) -> int:
    return (abs(v) & 0x7FFF) | (0x8000 if v >= 0 else 0)


def rd_frame(targets):
    body = b""
    for k in range(3):
        if k < len(targets):
            x, y, v, r = targets[k]
            body += struct.pack("<HHHH", _enc(x), _enc(y), _enc(v), r)
        else:
            body += bytes(8)
    f = HEADER + body + TAIL
    assert len(f) == FRAME_LEN
    return f


def test_rd03d_decode_and_signs():
    p = Rd03dParser()
    out = p.feed(rd_frame([(-300, 1200, -15, 360), (500, 2000, 0, 360)]), t=1.0)
    assert len(out) == 1 and len(out[0].tracks) == 2
    t0 = out[0].tracks[0]
    assert (t0.x_mm, t0.y_mm, t0.speed_cm_s) == (-300, 1200, -15)


def test_rd03d_resync_and_chunking():
    p = Rd03dParser()
    stream = b"\x01\x02\x03" + rd_frame([(100, 500, 5, 360)]) + rd_frame([])
    got = []
    for i in range(0, len(stream), 7):  # arbitrary chunking
        got += p.feed(stream[i:i + 7], t=0.0)
    assert len(got) == 2 and len(got[0].tracks) == 1 and got[1].tracks == ()


def test_rd03d_bad_tail_skipped():
    p = Rd03dParser()
    bad = bytearray(rd_frame([(1, 1, 1, 1)]))
    bad[-1] = 0x00
    out = p.feed(bytes(bad) + rd_frame([(2, 2, 2, 2)]), t=0.0)
    assert len(out) == 1 and p.stats.bad_frames >= 1


def test_vib_roundtrip_crc_and_drop_detection():
    rng = np.random.default_rng(0)
    s1 = rng.integers(-2000, 2000, (25, 2))
    s2 = rng.integers(-2000, 2000, (25, 2))
    p = VibFrameParser(scale=0.5)
    stream = encode_frame(10, 1_000_000, 500, s1) + encode_frame(12, 1_050_000, 500, s2)  # seq 11 missing
    pk = p.feed(stream)
    assert len(pk) == 2
    np.testing.assert_allclose(pk[0].data, s1 * 0.5)
    assert pk[0].t0 == 1.0 and p.stats.dropped_frames == 1


def test_vib_corrupt_frame_rejected_then_recovers():
    s = np.ones((10, 1), dtype=int)
    good = encode_frame(1, 0, 500, s)
    corrupt = bytearray(good)
    corrupt[10] ^= 0xFF
    p = VibFrameParser()
    out = p.feed(bytes(corrupt) + encode_frame(2, 20000, 500, s))
    assert len(out) == 1 and out[0].seq == 2 and p.stats.crc_errors >= 1


def test_vib_timestamp_unwrap():
    s = np.ones((5, 1), dtype=int)
    p = VibFrameParser()
    a = p.feed(encode_frame(1, 0xFFFFFF00, 500, s))[0].t0
    b = p.feed(encode_frame(2, 0x00000100, 500, s))[0].t0
    assert b > a
