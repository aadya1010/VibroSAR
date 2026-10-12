"""Record ESP32/ADXL345 vibration frames and print a health summary.

Usage: python tools/record_vibration.py --port COM4 --seconds 20 --out data/lab/vib_rest.npz
Sanity checks at rest (board flat): channel mean ~ +1 g on Z (~256 counts / 1.0 g with scale 0.0039),
dropped_frames = 0, crc_errors = 0.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.sensors.vib_serial import SerialVibrationSource  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--scale", type=float, default=0.0039, help="g per count (ADXL345 full-res)")
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--out", default="data/lab/vib_rec.npz")
    a = ap.parse_args()

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    chunks, t0s, fs = [], [], None
    start = time.monotonic()
    with SerialVibrationSource(a.port, a.baud, a.scale) as src:
        while time.monotonic() - start < a.seconds:
            for p in src.read():
                chunks.append(p.data)
                t0s.append(p.t0)
                fs = p.fs
            time.sleep(0.002)
        stats = src.stats
    if not chunks:
        print("no frames received: check port, baud, wiring, firmware")
        return
    data = np.vstack(chunks)
    np.savez(a.out, data=data, fs=fs, t0=t0s[0], synthetic=False)
    print(f"frames={stats.frames} crc_errors={stats.crc_errors} dropped_frames={stats.dropped_frames} fs={fs} Hz")
    print("mean (g):", np.round(data.mean(axis=0), 4), " std (g):", np.round(data.std(axis=0), 5))


if __name__ == "__main__":
    main()
