# VibroSAR (software core, simulator-first)

Hardware is ON HOLD. Everything here runs on SYNTHETIC data from `simulator/`. Results say nothing about real rubble.

For in-depth technical documentation, hardware protocols, and mathematical derivations, see [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md) and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Run
```
pip install -r requirements.txt
python -m pytest -q                        # 28 tests
python -m simulator.cli --scenario all --seed 0 --out data/synthetic
python scripts/run_benchmark.py --rebuild  # dataset -> train A/B -> reports/benchmark_report.md (~3-4 min)
```
## Pipeline
simulator -> `processing/` (phase demod, respiration features, NLMS + pilot check, matched-filter knocks)
-> `tracking/exclusion.py` (camera visible-person exclusion) -> `models/features.py` -> `models/baseline.py`
-> `fusion/state_machine.py` (NO_EVIDENCE / UNCERTAIN / POSSIBLE_SURVIVOR / HIGH_CONFIDENCE_CANDIDATE)

Detector A = needs raw-phase radar (not RD-03D). Detector B = RD-03D tracks + camera + ADXL345 (hardware-realistic).
`models/checkpoints/*.joblib` are trained on SYNTHETIC data only.

## Dashboard (SYNTHETIC replay)
```
python scripts/run_benchmark.py        # creates models/checkpoints (once)
uvicorn backend.api.app:app --port 8000
# open http://localhost:8000
```
