#!/usr/bin/env python3
"""The one R0/R1/R2 family: inverse-Wishart posterior-mean covariances of the corrected residuals.

R0 is global, R1 per camera, R2 per camera and spatial (16 neighbours, 0.4 m Gaussian
length scale, fixed by the method), all in the camera-to-query ray frame with the same broad
prior. Fitted on D_R and scored on D_dev; D_eval is never read. This is the covariance the
runtime fuses with and the planner inverts.

    python3 pipeline/fit_covariance.py --residuals FITS/corrected_residuals --output FITS/covariance
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import chi2


REPO = Path(__file__).resolve().parents[1]
CAMERAS = tuple(f"camera_{letter}" for letter in "ABCDE")
D = 2
PRIOR_STD_M = 10.0
PRIOR_STRENGTH = 2.5e-6
PRIOR_COVARIANCE = PRIOR_STD_M ** 2 * np.eye(D)
PRIOR_SCATTER = PRIOR_STRENGTH * PRIOR_COVARIANCE
# Spatial R2 constants, fixed by the method (not selected on data): 16 neighbours and a
# 0.4 m Gaussian length scale. The v8 top-up density K / (pi (2 l)^2) = 8 positions per m^2
# was derived from these same values. The grid below is reported as a sensitivity table
# on D_dev only; it never chooses the model.
R2_K_NEIGHBORS = 16
R2_LENGTH_SCALE_M = 0.4
K_GRID = (8, 16, 32)
LENGTH_GRID_M = (0.2, 0.3, 0.4, 0.6, 1.0, 1.4)
CHI95 = float(chi2.ppf(0.95, D))


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def grouped_moments(xy, position, residual, camera=None):
    groups = defaultdict(list)
    for index, key in enumerate(position):
        group = (str(key), str(camera[index])) if camera is not None else (str(key),)
        groups[group].append(index)
    points, cameras, moments = [], [], []
    for key in sorted(groups):
        use = np.asarray(groups[key], dtype=int)
        points.append(np.mean(xy[use], axis=0))
        cameras.append(key[-1] if camera is not None else "all")
        moments.append(np.mean(np.einsum("ni,nj->nij", residual[use], residual[use]), axis=0))
    return np.asarray(points), np.asarray(cameras), np.asarray(moments)


def equal_position(values, positions):
    return float(np.mean([np.mean(values[positions == key]) for key in np.unique(positions)]))


def metrics(residual, covariance, positions, cameras):
    inverse = np.linalg.inv(covariance)
    nis = np.einsum("ni,nij,nj->n", residual, inverse, residual)
    nll = 0.5 * (D * np.log(2.0 * np.pi) + np.linalg.slogdet(covariance)[1] + nis)
    area = np.pi * CHI95 * np.sqrt(np.linalg.det(covariance))
    result = {
        "observations": int(len(residual)),
        "positions": int(len(np.unique(positions))),
        "equal_position_mean_nll": equal_position(nll, positions),
        "equal_position_95pct_coverage": equal_position(nis <= CHI95, positions),
        "equal_position_mean_nis": equal_position(nis, positions),
        "equal_position_mean_95pct_ellipse_area_m2": equal_position(area, positions),
        "pooled_mean_nll": float(np.mean(nll)),
        "pooled_95pct_coverage": float(np.mean(nis <= CHI95)),
    }
    result["per_camera"] = {}
    for camera in CAMERAS:
        use = cameras == camera
        result["per_camera"][camera] = {
            "observations": int(np.sum(use)),
            "mean_nll": float(np.mean(nll[use])),
            "coverage_95pct": float(np.mean(nis[use] <= CHI95)),
            "mean_nis": float(np.mean(nis[use])),
        }
    return result


def local_statistics(train_xy, train_camera, train_moment, query_xy, query_camera, k, length):
    scatter = np.empty((len(query_xy), D, D))
    support = np.empty(len(query_xy))
    for camera in CAMERAS:
        use = train_camera == camera
        points, moments = train_xy[use], train_moment[use]
        for index in np.flatnonzero(query_camera == camera):
            distance2 = np.sum((points - query_xy[index]) ** 2, axis=1)
            count = min(k, len(points))
            selected = np.argpartition(distance2, count - 1)[:count]
            weight = np.exp(-0.5 * distance2[selected] / length ** 2)
            scatter[index] = np.einsum("n,nij->ij", weight, moments[selected])
            support[index] = float(np.sum(weight))
    return scatter, support


def posterior_mean(scatter, support):
    denominator = np.asarray(support, dtype=float) + PRIOR_STRENGTH
    if denominator.ndim == 0:
        return (scatter + PRIOR_SCATTER) / float(denominator)
    return (scatter + PRIOR_SCATTER) / denominator[..., None, None]


def matrix_summary(matrix):
    eigenvalues = np.linalg.eigvalsh(matrix)
    return {
        "covariance_ray_m2": matrix.tolist(),
        "principal_standard_deviation_cm": (100.0 * np.sqrt(eigenvalues)).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residuals", type=Path, required=True,
                        help="output folder of pipeline/corrected_residuals.py")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    residuals, final_output = args.residuals.resolve(), args.output.resolve()
    OUTPUT = final_output.with_name(final_output.name + ".incomplete")
    if final_output.exists() or OUTPUT.exists():
        raise FileExistsError(final_output if final_output.exists() else OUTPUT)
    residual_manifest_path = residuals / "manifest.json"
    residual_manifest = json.loads(residual_manifest_path.read_text(encoding="utf-8"))
    record = residual_manifest["artifacts"]["residuals"]
    SOURCE = residuals / record["path"]
    if sha256(SOURCE) != record["sha256"]:
        raise RuntimeError(f"corrected residuals hash drift: {SOURCE}")
    OUTPUT.mkdir(parents=True)
    with np.load(SOURCE, allow_pickle=False) as source:
        data = {name: np.asarray(source[name]) for name in source.files}
    fit = data["role"] == "D_R"
    dev = data["role"] == "D_dev"
    if np.any(data["role"] == "D_eval"):
        # Presence is expected. Values are never indexed below.
        pass
    residual = data["residual_ray_m"]
    dev_residual = residual[dev]
    dev_position = data["position_key"][dev].astype(str)
    dev_camera = data["camera"][dev].astype(str)

    _, _, global_moment = grouped_moments(
        data["truth_xy_m"][fit], data["position_key"][fit], residual[fit]
    )
    r0 = posterior_mean(np.sum(global_moment, axis=0), float(len(global_moment)))
    dev_r0 = np.broadcast_to(r0, (int(np.sum(dev)), D, D)).copy()

    r1 = {}
    r1_counts = {}
    for camera in CAMERAS:
        use = fit & (data["camera"] == camera)
        _, _, moment = grouped_moments(
            data["truth_xy_m"][use], data["position_key"][use], residual[use]
        )
        r1_counts[camera] = int(len(moment))
        r1[camera] = posterior_mean(np.sum(moment, axis=0), float(len(moment)))
    dev_r1 = np.stack([r1[camera] for camera in dev_camera])

    spatial_xy, spatial_camera, spatial_moment = grouped_moments(
        data["truth_xy_m"][fit], data["position_key"][fit], residual[fit], data["camera"][fit]
    )
    candidates = []
    cache = {}
    for k in K_GRID:
        for length in LENGTH_GRID_M:
            scatter, support = local_statistics(
                spatial_xy, spatial_camera, spatial_moment,
                data["corrected_xy_m"][dev], dev_camera, k, length,
            )
            covariance = posterior_mean(scatter, support)
            score = metrics(dev_residual, covariance, dev_position, dev_camera)
            candidate = {
                "k_neighbors": k,
                "length_scale_m": length,
                "equal_position_mean_nll": score["equal_position_mean_nll"],
                "equal_position_95pct_coverage": score["equal_position_95pct_coverage"],
            }
            candidates.append(candidate)
            cache[(k, length)] = (covariance, support, score)
    selected = next(row for row in candidates
                    if row["k_neighbors"] == R2_K_NEIGHBORS
                    and row["length_scale_m"] == R2_LENGTH_SCALE_M)
    dev_r2, dev_support, r2_metrics = cache[
        (selected["k_neighbors"], selected["length_scale_m"])
    ]

    model_path = OUTPUT / "models.npz"
    np.savez_compressed(
        model_path,
        camera_order=np.asarray(CAMERAS),
        prior_covariance_ray_m2=PRIOR_COVARIANCE,
        prior_strength=np.asarray([PRIOR_STRENGTH]),
        global_covariance_ray_m2=r0,
        per_camera_covariance_ray_m2=np.stack([r1[camera] for camera in CAMERAS]),
        spatial_reference_xy_m=spatial_xy,
        spatial_second_moment_ray_m2=spatial_moment,
        spatial_camera=spatial_camera,
        k_neighbors=np.asarray([selected["k_neighbors"]]),
        length_scale_m=np.asarray([selected["length_scale_m"]]),
    )
    report = {
        "schema": "final_bayesian_r012_fit.v1",
        "status": "fit_on_D_R_evaluated_on_D_dev",
        "roles_accessed": ["D_R", "D_dev"],
        "D_eval_accessed": False,
        "source": {"residuals_manifest": str(residual_manifest_path.relative_to(REPO)),
                   "residuals_manifest_sha256": sha256(residual_manifest_path),
                   "residuals_sha256": record["sha256"]},
        "parameterization": "posterior mean of proper inverse-Wishart covariance model",
        "runtime_frame": "camera-to-query ray frame",
        "position_weighting": "average second moment within physical position, then equal position weight",
        "prior": {
            "standard_deviation_m": PRIOR_STD_M,
            "covariance_ray_m2": PRIOR_COVARIANCE.tolist(),
            "strength": PRIOR_STRENGTH,
            "scatter_ray_m2": PRIOR_SCATTER.tolist(),
            "workspace_bounds_m": [-10.7, 10.7, -8.7, 8.7],
            "workspace_diagonal_m": float(np.hypot(21.4, 17.4)),
            "isotropic_95pct_radius_m": float(np.sqrt(CHI95) * PRIOR_STD_M),
        },
        "r2_constants": {"k_neighbors": R2_K_NEIGHBORS, "length_scale_m": R2_LENGTH_SCALE_M,
                         "rule": "fixed by the method; not selected on data"},
        "sensitivity_on_D_dev": {
            "candidate_grid": {"k_neighbors": K_GRID, "length_scale_m": LENGTH_GRID_M},
            "used_model": selected,
            "all_candidates": candidates,
        },
        "fit": {
            "R0": {"positions": int(len(global_moment)), **matrix_summary(r0)},
            "R1": {
                camera: {"positions": r1_counts[camera], **matrix_summary(r1[camera])}
                for camera in CAMERAS
            },
            "R2": {
                "support_positions": int(len(spatial_xy)),
                "support_by_camera": {
                    camera: int(np.sum(spatial_camera == camera)) for camera in CAMERAS
                },
                "zero_support_covariance_ray_m2": PRIOR_COVARIANCE.tolist(),
                "zero_support_principal_standard_deviation_m": [PRIOR_STD_M, PRIOR_STD_M],
                "D_dev_effective_support_quantiles": np.quantile(
                    dev_support, [0.0, 0.1, 0.5, 0.9, 1.0]
                ).tolist(),
                "D_dev_principal_standard_deviation_cm_quantiles": np.quantile(
                    100.0 * np.sqrt(np.linalg.eigvalsh(dev_r2)),
                    [0.0, 0.1, 0.5, 0.9, 1.0], axis=0,
                ).tolist(),
            },
        },
        "D_dev_metrics": {
            "R0": metrics(dev_residual, dev_r0, dev_position, dev_camera),
            "R1": metrics(dev_residual, dev_r1, dev_position, dev_camera),
            "R2": r2_metrics,
        },
        "artifacts": {"models": {"path": model_path.name, "sha256": sha256(model_path)}},
    }
    report_path = OUTPUT / "manifest.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUTPUT / ".complete").write_text(json.dumps({"manifest_sha256": sha256(report_path)}) + "\n")
    os.replace(OUTPUT, final_output)
    print(json.dumps({"R2": selected, "D_dev_metrics": report["D_dev_metrics"],
                      "manifest": str(final_output / report_path.name)}, indent=2))


if __name__ == "__main__":
    main()
