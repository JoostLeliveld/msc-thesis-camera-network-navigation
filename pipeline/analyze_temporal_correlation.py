#!/usr/bin/env python3
"""Between-frame correlation of the corrected measurement error along the driven campaign routes.

For every valid run in logs/thesis/analysis/runs.csv and every camera, the unique frames of
fusion_observations.csv (obs_repeat == 0, one row per obs_seq) give the residual
r = obs - gt_at_obs and its whitened form z = L^-1 r with the runtime covariance obs_cov = L L^T.
Every pair of frames of one camera in one run is binned by its time gap; per bin the Pearson
correlation of z (and of r) between the two frames is reported per world axis, with the median
ground-truth displacement between the two frames. Pairs whose ground-truth positions are within
STATIONARY_M are reported separately: there the correlation is temporal only, not spatial.
Writes logs/thesis/analysis/temporal_correlation.json.

    python3 pipeline/analyze_temporal_correlation.py
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ANALYSIS = REPO / "logs/thesis/final_campaign/analysis"
BINS_S = (0.1, 0.3, 0.5, 0.7, 0.9, 1.25, 1.75, 2.5, 3.5, 5.0, 8.0)
STATIONARY_M = 0.05


def frames(run: Path) -> dict[str, list[tuple]]:
    out, seen = defaultdict(list), set()
    with (run / "fusion_observations.csv").open(newline="", encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            key = (r["camera"], r["obs_seq"])
            if r["obs_repeat"] != "0" or key in seen:
                continue
            try:
                v = [float(r[k]) for k in ("obs_stamp", "obs_x", "obs_y", "gt_x_at_obs", "gt_y_at_obs",
                                            "obs_cov_xx", "obs_cov_xy", "obs_cov_yy")]
            except ValueError:
                continue
            if not all(math.isfinite(x) for x in v):
                continue
            seen.add(key)
            t, ox, oy, gx, gy, cxx, cxy, cyy = v
            res = np.array([ox - gx, oy - gy])
            z = np.linalg.solve(np.linalg.cholesky(np.array([[cxx, cxy], [cxy, cyy]])), res)
            out[r["camera"]].append((t, np.array([gx, gy]), res, z))
    return {c: sorted(v, key=lambda f: f[0]) for c, v in out.items()}


def corr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return [float(np.corrcoef(a[:, k], b[:, k])[0, 1]) for k in (0, 1)] if len(a) > 10 else None


def main() -> int:
    pairs = defaultdict(lambda: {"za": [], "zb": [], "ra": [], "rb": [], "moved": []})
    still = defaultdict(lambda: {"za": [], "zb": []})
    n_frames = 0
    with (ANALYSIS / "runs.csv").open(newline="", encoding="utf-8") as handle:
        runs = [REPO / r["run_dir"] for r in csv.DictReader(handle) if r.get("run_dir")]
    for run in runs:
        for cam, fs in frames(run).items():
            n_frames += len(fs)
            for i in range(len(fs)):
                for j in range(i + 1, len(fs)):
                    dt = fs[j][0] - fs[i][0]
                    if dt > BINS_S[-1]:
                        break
                    b = int(np.searchsorted(BINS_S, dt)) - 1
                    if b < 0:
                        continue
                    moved = float(np.hypot(*(fs[j][1] - fs[i][1])))
                    p = pairs[b]
                    p["za"].append(fs[i][3]); p["zb"].append(fs[j][3])
                    p["ra"].append(fs[i][2]); p["rb"].append(fs[j][2])
                    p["moved"].append(moved)
                    if moved < STATIONARY_M:
                        still[b]["za"].append(fs[i][3]); still[b]["zb"].append(fs[j][3])
    out = {"runs": len(runs), "frames": n_frames, "stationary_m": STATIONARY_M, "bins": []}
    for b in sorted(pairs):
        p, s = pairs[b], still[b]
        out["bins"].append({
            "gap_s": [BINS_S[b], BINS_S[b + 1]], "pairs": len(p["za"]),
            "median_displacement_m": float(np.median(p["moved"])),
            "corr_whitened_xy": corr(p["za"], p["zb"]), "corr_residual_xy": corr(p["ra"], p["rb"]),
            "stationary_pairs": len(s["za"]), "stationary_corr_whitened_xy": corr(s["za"], s["zb"]),
        })
    (ANALYSIS / "temporal_correlation.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"{out['runs']} runs, {n_frames} frames")
    for b in out["bins"]:
        cw = b["corr_whitened_xy"]; sc = b["stationary_corr_whitened_xy"]
        print(f"gap {b['gap_s'][0]:4.2f}-{b['gap_s'][1]:4.2f} s  pairs {b['pairs']:6d}  moved {b['median_displacement_m']:5.2f} m  "
              f"corr z {cw[0]:+.2f} {cw[1]:+.2f}  | stationary n={b['stationary_pairs']:5d} "
              + (f"corr z {sc[0]:+.2f} {sc[1]:+.2f}" if sc else "-"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
