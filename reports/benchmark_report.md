# VibroSAR benchmark (SYNTHETIC data only)

Train seeds 0-9, calibration/threshold seeds 10-11, test seeds 12-15. Threshold = max recall with FPR <= 5% on validation.

## Overall (test windows)

| detector | precision | recall | f1 | fpr | fnr | auroc | ece | n |
|---|---|---|---|---|---|---|---|---|
| A_full_sim | 0.995 | 0.856 | 0.920 | 0.007 | 0.144 | 0.988 | 0.028 | 360.000 |
| B_hardware_like | 0.985 | 0.903 | 0.942 | 0.021 | 0.097 | 0.979 | 0.025 | 360.000 |
| naive_breathing_anywhere | 0.704 | 0.968 | 0.815 | 0.611 | 0.032 | 0.678 | 0.264 | 360.000 |


## Fraction of windows flagged positive, per scenario (want ~0 for 'none', ~1 for 'survivor')

| scenario | truth | naive | A_full_sim | B_hardware_like |
|---|---|---|---|---|
| no_people | none | 0.00 | 0.00 | 0.00 |
| one_worker | none | 0.67 | 0.00 | 0.00 |
| multi_workers | none | 1.00 | 0.03 | 0.00 |
| survivor_alone | survivor | 1.00 | 1.00 | 1.00 |
| worker_survivor | survivor | 0.89 | 0.44 | 0.72 |
| multi_workers_survivor | survivor | 1.00 | 0.78 | 0.78 |
| worker_machinery | none | 0.78 | 0.00 | 0.08 |
| survivor_machinery | survivor | 1.00 | 0.94 | 1.00 |
| worker_survivor_machinery | survivor | 0.92 | 0.97 | 0.92 |
| multi_workers_survivor_machinery | survivor | 1.00 | 1.00 | 1.00 |


## Recording-level states, detector A_full_sim

| scenario | truth | reached_possible | reached_high | t_first_possible_s |
|---|---|---|---|---|
| multi_workers | 0 | 0.00 | 0.00 | nan |
| multi_workers_survivor | 1 | 0.75 | 0.75 | 40.00 |
| multi_workers_survivor_machinery | 1 | 1.00 | 0.00 | 40.00 |
| no_people | 0 | 0.00 | 0.00 | nan |
| one_worker | 0 | 0.00 | 0.00 | nan |
| survivor_alone | 1 | 1.00 | 1.00 | 40.00 |
| survivor_machinery | 1 | 1.00 | 0.00 | 40.00 |
| worker_machinery | 0 | 0.00 | 0.00 | nan |
| worker_survivor | 1 | 0.50 | 0.50 | 40.00 |
| worker_survivor_machinery | 1 | 1.00 | 0.00 | 40.00 |


Localization (A, true-positive windows with a radar candidate): n=151, median error 0.04 m, p90 1.61 m (range bin = 0.1 m)

## Recording-level states, detector B_hardware_like

| scenario | truth | reached_possible | reached_high | t_first_possible_s |
|---|---|---|---|---|
| multi_workers | 0 | 0.00 | 0.00 | nan |
| multi_workers_survivor | 1 | 0.75 | 0.75 | 40.00 |
| multi_workers_survivor_machinery | 1 | 1.00 | 0.00 | 40.00 |
| no_people | 0 | 0.00 | 0.00 | nan |
| one_worker | 0 | 0.00 | 0.00 | nan |
| survivor_alone | 1 | 1.00 | 1.00 | 40.00 |
| survivor_machinery | 1 | 1.00 | 0.00 | 40.00 |
| worker_machinery | 0 | 0.25 | 0.00 | 90.00 |
| worker_survivor | 1 | 0.75 | 0.75 | 40.00 |
| worker_survivor_machinery | 1 | 1.00 | 0.00 | 40.00 |


## Ablation (hardware-realistic evidence). Radar-track features carry much of B's score ONLY because the simulator lets a 24 GHz tracker partly see through light cover (10 dB/m). Treat vibration-only as the conservative estimate.

| model | recall | fpr | auroc |
|---|---|---|---|
| B full (vib + trk + camera) | 0.903 | 0.021 | 0.979 |
| B minus radar tracks | 0.815 | 0.097 | 0.902 |
| vibration only | 0.593 | 0.014 | 0.853 |
| vibration, no interference/pilot feats | 0.616 | 0.007 | 0.853 |


## Cover-attenuation sweep (train at 10 dB/m, test at other values; 24 GHz radar assumption)

| atten_db_per_m | A_full_sim TPR | A_full_sim FPR | B_hardware_like TPR | B_hardware_like FPR |
|---|---|---|---|---|
| 5.00 | 0.85 | 0.00 | 0.88 | 0.03 |
| 10.00 | 0.80 | 0.00 | 0.90 | 0.03 |
| 20.00 | 0.79 | 0.00 | 0.83 | 0.03 |
| 30.00 | 0.75 | 0.00 | 0.71 | 0.03 |
| 40.00 | 0.74 | 0.00 | 0.69 | 0.03 |

