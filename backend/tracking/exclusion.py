"""Visible-person exclusion map over radar range bins (time-resolved).

Source: camera person detections mapped to world (x, y) (radar at origin).
For each 1 s step in the analysis window, a range bin is 'occupied' if a detection in +-half_s lies
  * PERSON zone : within +-margin_m of the bin, or
  * GHOST zone  : the bin is between ghost_lo_m and ghost_hi_m BEHIND the person (multipath ghost).
A bin is EXCLUDED for the window if it is occupied for >= frac_thr of the steps.
=> a STATIONARY worker excludes its bin (and ghost) for the whole window, while a WALKING worker
   only briefly crosses each bin (and large motion is rejected separately by the motion gate).

The ghost offset is environment dependent; 0.5-1.1 m matches the simulator (ghost at +0.8 m) and
MUST be calibrated on real recordings. Known limitation: IQ radar data is range-only, so a buried
survivor at the same range as a stationary visible worker (different bearing) is excluded too.
"""
from __future__ import annotations

import numpy as np


def exclusion_mask(cam_t: np.ndarray, cam_xy: np.ndarray, t0: float, t1: float, r_axis: np.ndarray,
                   margin_m: float = 0.35, ghost_lo_m: float = 0.5, ghost_hi_m: float = 1.1,
                   step_s: float = 1.0, half_s: float = 1.0, frac_thr: float = 0.5) -> np.ndarray:
    """Boolean mask over range bins (True = excluded)."""
    steps = np.arange(t0 + step_s / 2, t1, step_s)
    occupied = np.zeros((len(steps), len(r_axis)), bool)
    for k, s in enumerate(steps):
        sel = (cam_t >= s - half_s) & (cam_t <= s + half_s)
        xy = cam_xy[sel].reshape(-1, 2)
        xy = xy[~np.isnan(xy).any(axis=1)]
        if len(xy) == 0:
            continue
        d = r_axis[:, None] - np.hypot(xy[:, 0], xy[:, 1])[None, :]
        occupied[k] = ((np.abs(d) <= margin_m) | ((d >= ghost_lo_m) & (d <= ghost_hi_m))).any(axis=1)
    return occupied.mean(axis=0) >= frac_thr
