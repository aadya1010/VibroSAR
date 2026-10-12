# VibroSAR architecture (Phase 1)

- `backend/core`: types (Candidate, packets, enums), clock/ring buffer, config.
- `backend/sensors`: `RadarSource` / `VibrationSource` / `PersonSource` interfaces.
  - RD-03D = Mode B (tracks only, `has_phase=False`) -> respiration branch auto-disabled.
  - Vibration framing protocol documented in `vib_serial.py`.
- Next: simulator (Phase 2), DSP (Phase 3), fusion (Phase 4).

Rules: synthetic data always flagged; depth stays None unless calibrated;
confidence is a model score, not a validated survival probability.

## Phase 2: simulator (all output SYNTHETIC)
- `simulator/scenario.py`: 10 benchmark scenarios via `make_scenario(name, seed, **overrides)`.
- `simulator/radar_model.py`: Mode A IQ (slow-time x range-bin, phase = 4*pi*R/lambda, clutter, DC/IQ imbalance,
  rubble attenuation, multipath ghosts, aliased machine signature) and Mode B emulated tracks.
- `simulator/vib_model.py`: ADXL345-like channels (1 g DC, quantized): machinery, footsteps, knocks, multipath propagation.
- `simulator/simulate.py`: `run_scenario(cfg) -> SimResult` with streams + ground truth; `.save()/.load()`.
- `backend/sensors/sim_sources.py`: `SimRadarSource`, `SimVibrationSource` (same interfaces as hardware).
- CLI: `python -m simulator.cli --scenario all --seed 0 --out data/synthetic`
Known simplification: single-bin-per-person range model, no true electromagnetic propagation; all parameters are assumptions.

## Phase 3-5 (software-first)
- `processing/`: `phase.py` (circle-fit arctangent demod), `respiration.py` (micro-motion features; large-motion and amplitude
  non-stationarity gates), `adaptive.py` (NLMS with gating + pilot-injection safety test), `vibration.py` (matched-filter
  knock/footstep detection, knock-triplet pattern score, machine peakiness/coherence).
- `tracking/exclusion.py`: time-resolved camera exclusion map; ghost duplicates rejected by waveform correlation.
- `models/features.py`, `dataset.py`, `baseline.py`; `fusion/state_machine.py`; `scripts/run_benchmark.py`.
Known weaknesses: range-only radar cannot separate a survivor from a stationary worker at the same range; machinery aliasing
can look like respiration; knock/step templates are simulator assumptions; small test set (4 seeds).
