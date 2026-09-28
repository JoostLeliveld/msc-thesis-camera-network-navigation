#!/usr/bin/env python3
"""Calibration of the runtime covariances along the driven campaign routes, per condition.

For every valid run in logs/thesis/analysis/runs.csv:
  fused   one row per fusion decision with at least two admitted cameras (as in the static
          fusion audit): fused estimate vs ground truth at the fused stamp, fused covariance
          (fusion_observations.csv);
  belief  the estimator (est_x/y, est_cov_*) after the first command vs ground truth interpolated
          at the planner belief stamp (experiment.csv, ground_truth_pose.csv). state_cov_* is not
          the estimator covariance (it saturates), so it is not used.
Containment is the fraction with squared Mahalanobis distance below the 2-D chi-square 95 %
quantile; NIS is its mean. Pooled over runs, and run-balanced (mean of per-run values).
Writes logs/thesis/analysis/runtime_coverage.json.

    python3 figures/make_runtime_coverage.py
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
CHI2_95 = -2.0 * math.log(0.05)  # 2-D chi-square 95 % quantile, 5.991


def mahalanobis2(dx, dy, cxx, cxy, cyy):
    det = cxx * cyy - cxy * cxy
    return (cyy * dx * dx - 2 * cxy * dx * dy + cxx * dy * dy) / det


def fused(run: Path) -> list[float]:
    by_decision = {}
    with (run / "fusion_observations.csv").open(newline="", encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            if int(r["n_used"]) < 2 or r["decision_seq"] in by_decision:
                continue
            v = [float(r[k]) for k in ("fused_x", "fused_y", "gt_x_at_fused", "gt_y_at_fused",
                                        "fused_cov_xx", "fused_cov_xy", "fused_cov_yy")]
            if all(math.isfinite(x) for x in v):
                by_decision[r["decision_seq"]] = mahalanobis2(v[0] - v[2], v[1] - v[3], *v[4:])
    return list(by_decision.values())


def belief(run: Path) -> list[float]:
    first_cmd = float(json.loads((run / "run_summary.json").read_text())["first_cmd_stamp"])
    with (run / "ground_truth_pose.csv").open(newline="", encoding="utf-8") as handle:
        gt = np.array([(float(r["stamp_s"]), float(r["x"]), float(r["y"])) for r in csv.DictReader(handle)])
    out, seen = [], set()
    with (run / "experiment.csv").open(newline="", encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            try:
                t = float(r["planner_belief_stamp"])
                v = [float(r[k]) for k in ("est_x", "est_y", "est_cov_xx", "est_cov_xy", "est_cov_yy")]
            except (KeyError, ValueError):
                continue
            if not (math.isfinite(t) and all(map(math.isfinite, v))) or t < first_cmd or t in seen:
                continue
            if t < gt[0, 0] or t > gt[-1, 0]:
                continue
            seen.add(t)
            gx, gy = np.interp(t, gt[:, 0], gt[:, 1]), np.interp(t, gt[:, 0], gt[:, 2])
            out.append(mahalanobis2(v[0] - gx, v[1] - gy, *v[2:]))
    return out


def stats(per_run: list[list[float]]) -> dict:
    pooled = np.concatenate([np.asarray(d) for d in per_run if d]) if any(per_run) else np.array([])
    runs = [np.asarray(d) for d in per_run if d]
    return {
        "runs": len(runs), "samples": int(pooled.size),
        "pooled_coverage_95": float(np.mean(pooled < CHI2_95)) if pooled.size else None,
        "pooled_mean_nis": float(np.mean(pooled)) if pooled.size else None,
        "run_balanced_coverage_95": float(np.mean([np.mean(d < CHI2_95) for d in runs])) if runs else None,
        "run_balanced_mean_nis": float(np.mean([np.mean(d) for d in runs])) if runs else None,
    }


def main() -> int:
    groups = defaultdict(lambda: {"fused": [], "belief": []})
    with (ANALYSIS / "runs.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not row.get("run_dir"):
                continue
            run = REPO / row["run_dir"]
            f, b = fused(run), belief(run)
            for key in (row["condition"], row["model"]):
                groups[key]["fused"].append(f)
                groups[key]["belief"].append(b)
    out = {"chi2_95": CHI2_95,
           "rule": "fused: decisions with >= 2 admitted cameras; belief: state after first command",
           "groups": {k: {kind: stats(v[kind]) for kind in ("fused", "belief")} for k, v in sorted(groups.items())}}
    (ANALYSIS / "runtime_coverage.json").write_text(json.dumps(out, indent=1) + "\n")
    for k, v in out["groups"].items():
        print(f"{k:22s} fused C95 {v['fused']['pooled_coverage_95']:.4f} NIS {v['fused']['pooled_mean_nis']:.2f} "
              f"n={v['fused']['samples']:6d} | belief C95 {v['belief']['pooled_coverage_95']:.4f} "
              f"NIS {v['belief']['pooled_mean_nis']:.2f} n={v['belief']['samples']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
