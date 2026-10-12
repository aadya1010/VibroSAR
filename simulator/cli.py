"""Generate synthetic datasets:  python -m simulator.cli --scenario all --seed 0 --out data/synthetic"""
import argparse
import logging
from pathlib import Path

from simulator.scenario import SCENARIO_NAMES, make_scenario
from simulator.simulate import run_scenario


def main() -> None:
    ap = argparse.ArgumentParser(description="VibroSAR simulator (all output is SYNTHETIC)")
    ap.add_argument("--scenario", default="all", help=f"one of {SCENARIO_NAMES} or 'all'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--out", default="data/synthetic")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name in (SCENARIO_NAMES if a.scenario == "all" else [a.scenario]):
        res = run_scenario(make_scenario(name, a.seed, duration_s=a.duration))
        path = Path(a.out) / f"SYNTH_{name}_seed{a.seed}.npz"
        res.save(path)
        print(f"{name:36s} persons={len(res.pids)} machines={len(res.cfg.machines)} "
              f"knocks={len(res.knock_times)} -> {path}")


if __name__ == "__main__":
    main()
