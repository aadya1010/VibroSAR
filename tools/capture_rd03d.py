"""Record raw RD-03D serial bytes and print a decoded summary.

Usage:  python tools/capture_rd03d.py --port COM3 --seconds 60 --out data/lab/rd03d_dump.bin
Send us the printed summary + first ~200 bytes of hex to verify the frame format.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import serial  # noqa: E402

from backend.sensors.radar_tracks import Rd03dParser  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=256000)
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--out", default="data/lab/rd03d_dump.bin")
    a = ap.parse_args()

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    parser, raw, n_tracks, t0 = Rd03dParser(), bytearray(), 0, time.monotonic()
    with serial.Serial(a.port, a.baud, timeout=0.1) as s, open(a.out, "wb") as f:
        while time.monotonic() - t0 < a.seconds:
            d = s.read(256)
            if not d:
                continue
            f.write(d)
            if len(raw) < 200:
                raw.extend(d[: 200 - len(raw)])
            for p in parser.feed(d, time.monotonic() - t0):
                n_tracks += len(p.tracks)
    dur = time.monotonic() - t0
    print(f"duration={dur:.1f}s frames={parser.stats.frames} bad_frames={parser.stats.bad_frames} "
          f"fps={parser.stats.frames / dur:.1f} track_observations={n_tracks}")
    print("first bytes:", raw.hex(" "))


if __name__ == "__main__":
    main()
