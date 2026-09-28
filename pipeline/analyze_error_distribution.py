#!/usr/bin/env python3
"""Shape and structure of the corrected measurement error on the test set (final audit).

Reads logs/thesis/final_audit/evaluated.jsonl (squared Mahalanobis distance per admitted
observation and model) and logs/thesis/final_audit_fusion/evaluated_admitted.jsonl (corrected
residual vector and every model's covariance). Writes logs/thesis/analysis/error_distribution.json.

  1. NIS distribution: quantiles of d^2 against chi-square(2), exceedance of the 99.9 % bound,
     Kolmogorov-Smirnov distance, and 95 % bootstrap intervals over positions for C95 and mean NIS.
  2. Shape of the whitened residual z = L^-1 r (R = L L^T): kurtosis per axis (Gaussian 3).
  3. How systematic: share of residual variance shared by all headings of one (position, camera)
     (intraclass correlation), and correlation of that shared part between nearby positions.
  4. Cross-camera correlation: correlation of whitened residuals of two cameras in the same
     position-heading batch, per camera pair.
  5. Heading: C95 and mean NIS per heading index.

    python3 pipeline/analyze_error_distribution.py
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
T = REPO / "logs/thesis"
OUT = T / "analysis/error_distribution.json"
MODELS = {"global": "R0_global_full", "per_camera": "R1_per_camera_full", "spatial": "R2_spatial_full",
          "rproj": "Rproj"}
LEVELS = (0.5, 0.9, 0.95, 0.99, 0.999)
BOOTSTRAP, SEED = 2000, 20260926
NEAR_M = 0.5


def chi2_2_quantile(p: float) -> float:
    return -2.0 * math.log(1.0 - p)


def chi2_2_cdf(x):
    return 1.0 - np.exp(-np.asarray(x) / 2.0)


def load():
    audit = [json.loads(line) for line in (T / "final_audit/evaluated.jsonl").open()]
    audit = [r for r in audit if r["outcome"] == "admitted"]
    fusion = [json.loads(line) for line in (T / "final_audit_fusion/evaluated_admitted.jsonl").open()]
    return audit, fusion


def nis_block(audit, model_key, rng):
    d2 = np.array([r["mahalanobis_d2"][model_key] for r in audit])
    pos = np.array([r["position_key"] for r in audit])
    keys = sorted(set(pos))
    per = {k: d2[pos == k] for k in keys}

    def equal_position(sample):
        return (float(np.mean([np.mean(per[k] < chi2_2_quantile(0.95)) for k in sample])),
                float(np.mean([np.mean(per[k]) for k in sample])))

    c95, nis = equal_position(keys)
    boots = np.array([equal_position(rng.choice(keys, len(keys))) for _ in range(BOOTSTRAP)])
    xs = np.sort(d2)
    ks = float(np.max(np.abs(np.arange(1, len(xs) + 1) / len(xs) - chi2_2_cdf(xs))))
    return {
        "observations": int(d2.size), "positions": len(keys),
        "equal_position_c95": c95, "equal_position_c95_ci95": np.percentile(boots[:, 0], [2.5, 97.5]).tolist(),
        "equal_position_mean_nis": nis, "equal_position_mean_nis_ci95": np.percentile(boots[:, 1], [2.5, 97.5]).tolist(),
        "pooled_coverage": {str(p): float(np.mean(d2 < chi2_2_quantile(p))) for p in LEVELS},
        "pooled_d2_quantiles": {str(p): float(np.quantile(d2, p)) for p in LEVELS},
        "chi2_2_quantiles": {str(p): chi2_2_quantile(p) for p in LEVELS},
        "exceed_99_9_fraction": float(np.mean(d2 > chi2_2_quantile(0.999))),
        "ks_distance_vs_chi2_2": ks,
    }


def whiten(r, cov):
    return np.linalg.solve(np.linalg.cholesky(np.asarray(cov)), np.asarray(r))


def residuals(fusion, model):
    out = []
    for r in fusion:
        res = np.subtract(r["corrected_xy_m"], r["truth_xy_m"])
        out.append({"pos": r["position_key"], "yaw": r["yaw_idx"], "cam": r["camera_id"], "r": res,
                    "z": whiten(res, r["covariance_m2"][model]), "truth": np.asarray(r["truth_xy_m"])})
    return out


def kurtosis(x):
    x = np.asarray(x) - np.mean(x)
    return float(np.mean(x ** 4) / np.mean(x ** 2) ** 2)


def systematic(res):
    """Intraclass correlation of the residual over the headings of one (position, camera)."""
    groups = defaultdict(list)
    for o in res:
        groups[(o["pos"], o["cam"])].append(o["r"])
    groups = {k: np.array(v) for k, v in groups.items() if len(v) >= 2}
    within = np.mean([np.mean(np.sum((v - v.mean(0)) ** 2, axis=1)) * len(v) / (len(v) - 1) for v in groups.values()])
    total = np.mean(np.sum((np.concatenate(list(groups.values())) - np.concatenate(list(groups.values())).mean(0)) ** 2, axis=1))
    means = {k: v.mean(0) for k, v in groups.items()}
    truth = {}
    for o in res:
        truth[o["pos"]] = o["truth"]
    # correlation of the per-(position, camera) mean residual between nearby positions, same camera
    a, b = [], []
    by_cam = defaultdict(list)
    for (pos, cam), m in means.items():
        by_cam[cam].append((truth[pos], m))
    for items in by_cam.values():
        for (p1, m1), (p2, m2) in combinations(items, 2):
            if np.hypot(*(p1 - p2)) <= NEAR_M:
                a.append(m1)
                b.append(m2)
    a, b = np.array(a), np.array(b)
    near = [float(np.corrcoef(a[:, k], b[:, k])[0, 1]) for k in (0, 1)] if len(a) > 3 else None
    return {"groups": len(groups), "within_heading_variance_m2": float(within), "total_variance_m2": float(total),
            "shared_fraction_icc": float(1.0 - within / total),
            "near_pairs": int(len(a)), "near_radius_m": NEAR_M, "near_mean_residual_corr_xy": near}


def cross_camera(res):
    batches = defaultdict(dict)
    for o in res:
        batches[(o["pos"], o["yaw"])][o["cam"]] = o["z"]
    pairs = defaultdict(list)
    for b in batches.values():
        for c1, c2 in combinations(sorted(b), 2):
            pairs[(c1, c2)].append(np.concatenate([b[c1], b[c2]]))
    out, pooled = {}, []
    for (c1, c2), rows in sorted(pairs.items()):
        rows = np.array(rows)
        pooled.append(rows)
        if len(rows) < 10:
            out[f"{c1}-{c2}"] = {"batches": len(rows)}
            continue
        c = np.corrcoef(rows.T)
        out[f"{c1}-{c2}"] = {"batches": len(rows), "corr_xx": float(c[0, 2]), "corr_yy": float(c[1, 3]),
                             "corr_xy": float(c[0, 3]), "corr_yx": float(c[1, 2])}
    rows = np.concatenate(pooled)
    c = np.corrcoef(rows.T)
    out["pooled"] = {"pairs": len(rows), "corr_xx": float(c[0, 2]), "corr_yy": float(c[1, 3])}
    return out


def by_heading(audit, model_key):
    out = {}
    for yaw in sorted({r["yaw_idx"] for r in audit}):
        d2 = np.array([r["mahalanobis_d2"][model_key] for r in audit if r["yaw_idx"] == yaw])
        out[str(yaw)] = {"observations": int(d2.size), "c95": float(np.mean(d2 < chi2_2_quantile(0.95))),
                         "mean_nis": float(np.mean(d2))}
    return out


def main() -> int:
    rng = np.random.default_rng(SEED)
    audit, fusion = load()
    out = {"population": "final audit (D_test), admitted observations", "nis": {}, "whitened_kurtosis": {},
           "per_camera_spatial": {}, "heading": {}}
    for name, key in MODELS.items():
        out["nis"][name] = nis_block(audit, key, rng)
        out["heading"][name] = by_heading(audit, key)
    for cam in sorted({r["camera_id"] for r in audit}):
        sub = [r for r in audit if r["camera_id"] == cam]
        d2 = np.array([r["mahalanobis_d2"]["R2_spatial_full"] for r in sub])
        out["per_camera_spatial"][cam] = {"observations": int(d2.size), "c95": float(np.mean(d2 < chi2_2_quantile(0.95))),
                                          "mean_nis": float(np.mean(d2))}
    for name in ("global", "per_camera", "spatial"):
        z = np.array([o["z"] for o in residuals(fusion, name)])
        out["whitened_kurtosis"][name] = [kurtosis(z[:, 0]), kurtosis(z[:, 1])]
    spatial = residuals(fusion, "spatial")
    out["systematic"] = systematic(spatial)
    out["cross_camera_spatial_whitened"] = cross_camera(spatial)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
