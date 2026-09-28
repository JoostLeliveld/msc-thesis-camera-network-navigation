#!/usr/bin/env python3
"""D_dev evaluation of the runtime covariances R0/R1/R2 and of the R_proj baseline.

R0/R1/R2 are the models pipeline/fit_covariance.py fitted (the ones the runtime fuses with);
R2 is rebuilt here exactly as the runtime queries it. R_proj propagates sigma_px^2 I2 through
the pixel-to-ground Jacobian; sigma_px is fitted on D_R. Per-position scores and a figure.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src/reliability"), str(REPO / "src/unav_common"), str(REPO)]
from pipeline.fit_covariance import CAMERAS, local_statistics, posterior_mean  # noqa: E402
from pipeline.dataset import load_rows  # noqa: E402
from reliability.contracts import CameraObservation  # noqa: E402
from reliability.projection import (  # noqa: E402
    camera_model_from_world, project_observation_to_world_with_covariance,
)


EIGENVALUE_FLOOR_M2 = 1.0e-6


def psd(matrix: np.ndarray) -> np.ndarray:
    eig, vec = np.linalg.eigh(0.5 * (matrix + matrix.T))
    return (vec * np.maximum(eig, EIGENVALUE_FLOOR_M2)) @ vec.T


def metrics(residual, covariance, position):
    nll, nis = [], []
    for value, matrix in zip(residual, covariance):
        matrix = psd(matrix); inverse = np.linalg.inv(matrix)
        score = float(value @ inverse @ value)
        nis.append(score)
        nll.append(0.5 * (math.log(np.linalg.det(matrix)) + score + 2 * math.log(2 * math.pi)))
    nll, nis = np.asarray(nll), np.asarray(nis)
    by_position = []
    for key in sorted(set(position.tolist())):
        use = position == key
        by_position.append((float(nll[use].mean()), float(nis[use].mean()), float((nis[use] <= 5.991).mean())))
    return {
        "observations": len(residual), "positions": len(by_position),
        "equal_position_mean_nll": float(np.mean([x[0] for x in by_position])),
        "equal_position_mean_nis": float(np.mean([x[1] for x in by_position])),
        "equal_position_95pct_coverage": float(np.mean([x[2] for x in by_position])),
        "pooled_mean_nll": float(nll.mean()), "pooled_mean_nis": float(nis.mean()),
        "pooled_95pct_coverage": float((nis <= 5.991).mean()),
        "mean_logdet": float(np.mean([math.log(np.linalg.det(psd(x))) for x in covariance])),
    }


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def read_verified_artifact(directory: Path, key: str) -> tuple[dict, Path]:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest["artifacts"][key]
    path = directory / record["path"]
    if sha256(path) != record["sha256"]:
        raise RuntimeError(f"artifact hash drift: {path}")
    return manifest, path


def unit_pixel_covariances(data: dict[str, np.ndarray]) -> np.ndarray:
    rows = load_rows()
    world = REPO / "src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf"
    models = {camera: camera_model_from_world(world, include_name=next(
        row["camera_model"] for row in rows if row["camera_id"] == camera
    )) for camera in CAMERAS}
    output = np.empty((len(data["camera"]), 2, 2), dtype=float)
    for index, (camera, box) in enumerate(zip(data["camera"], data["bbox_xyxy"], strict=True)):
        bottom = (0.5 * (float(box[0]) + float(box[2])), float(box[3]))
        observation = CameraObservation(
            camera_id=str(camera), timestamp_s=0.0, pixel_uv=bottom,
            detection_valid=True, detector_score=1.0,
            conditional_cov_uv=((1.0, 0.0), (0.0, 1.0)),
        )
        projected = project_observation_to_world_with_covariance(observation, models[str(camera)])
        if projected is None:
            raise RuntimeError(f"Rproj projection failed at admitted observation {index}")
        output[index] = np.asarray(projected[1], dtype=float)
    return output


def fit_sigma(position: np.ndarray, residual: np.ndarray, unit: np.ndarray) -> float:
    squared = np.einsum("ni,nij,nj->n", residual, np.linalg.inv(unit), residual)
    by_position = [float(squared[position == key].mean()) for key in sorted(set(position.tolist()))]
    sigma2 = float(np.mean(by_position) / 2.0)
    if not math.isfinite(sigma2) or sigma2 <= 0.0:
        raise RuntimeError("invalid Rproj sigma")
    return math.sqrt(sigma2)


def score_arrays(residual: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nll, nis = np.empty(len(residual)), np.empty(len(residual))
    for i, (value, matrix) in enumerate(zip(residual, covariance, strict=True)):
        matrix = psd(matrix); inverse = np.linalg.inv(matrix)
        nis[i] = float(value @ inverse @ value)
        nll[i] = 0.5 * (math.log(np.linalg.det(matrix)) + nis[i] + 2 * math.log(2 * math.pi))
    return nll, nis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correction", type=Path, required=True)
    parser.add_argument("--covariance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    correction, covariance, output = (p.resolve() for p in (args.correction, args.covariance, args.output))
    staging = output.with_name(output.name + ".incomplete")
    if output.exists() or staging.exists():
        raise FileExistsError(output if output.exists() else staging)
    staging.mkdir(parents=True)

    correction_manifest, predictions_path = read_verified_artifact(correction, "predictions")
    covariance_manifest, models_path = read_verified_artifact(covariance, "models")
    if correction_manifest.get("final_audit_accessed") or covariance_manifest.get("final_audit_accessed"):
        raise RuntimeError("source artifact opened final_audit")
    with np.load(predictions_path, allow_pickle=False) as source:
        data = {name: np.asarray(source[name]) for name in source.files}
    with np.load(models_path, allow_pickle=False) as source:
        model = {name: np.asarray(source[name]) for name in source.files}

    unit = unit_pixel_covariances(data)
    fit, dev = data["role"] == "D_R", data["role"] == "D_dev"
    sigma_px = fit_sigma(data["position_key"][fit], data["residual_world_m"][fit], unit[fit])
    r0 = model["global_covariance_ray_m2"]
    r1 = {str(camera): matrix for camera, matrix in zip(
        model["camera_order"], model["per_camera_covariance_ray_m2"], strict=True)}

    # Rebuild the corrected-observation ray residual used by the frozen covariance fit.
    rows = load_rows(); world = REPO / "src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf"
    camera_xy = {}
    for camera in CAMERAS:
        include = next(row["camera_model"] for row in rows if row["camera_id"] == camera)
        camera_xy[camera] = np.asarray(camera_model_from_world(world, include_name=include).cam_pos[:2])
    residual_ray = np.empty_like(data["residual_world_m"], dtype=float)
    for i, (camera, point, residual) in enumerate(zip(
            data["camera"], data["corrected_xy_m"], data["residual_world_m"], strict=True)):
        along = point - camera_xy[str(camera)]; along /= np.linalg.norm(along)
        residual_ray[i] = np.column_stack((along, [-along[1], along[0]])).T @ residual

    r0_cov = np.repeat(r0[None], int(dev.sum()), axis=0)
    r1_cov = np.stack([r1[str(camera)] for camera in data["camera"][dev]])
    r2_cov = posterior_mean(*local_statistics(
        model["spatial_reference_xy_m"], model["spatial_camera"].astype(str),
        model["spatial_second_moment_ray_m2"], data["corrected_xy_m"][dev],
        data["camera"][dev].astype(str), int(model["k_neighbors"][0]),
        float(model["length_scale_m"][0]),
    ))
    rproj_cov = sigma_px ** 2 * unit[dev]
    covariance_by_name = {
        "R0_global_full": (residual_ray[dev], r0_cov),
        "R1_per_camera_full": (residual_ray[dev], r1_cov),
        "R2_spatial_residual": (residual_ray[dev], r2_cov),
        "Rproj_homography_pixel": (data["residual_world_m"][dev], rproj_cov),
    }
    scores = {name: metrics(residual, cov, data["position_key"][dev])
              for name, (residual, cov) in covariance_by_name.items()}

    per_position_path = staging / "ddev_per_position.csv"
    dev_position = data["position_key"][dev]
    with per_position_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "position_key", "observations", "correction_mean_error_m",
            "correction_rmse_m", *(f"{name}_mean_nll" for name in covariance_by_name),
            *(f"{name}_mean_nis" for name in covariance_by_name),
        ])
        writer.writeheader()
        score_cache = {name: score_arrays(*values) for name, values in covariance_by_name.items()}
        error = np.linalg.norm(data["residual_world_m"][dev], axis=1)
        for key in sorted(set(dev_position.tolist())):
            use = dev_position == key
            row = {"position_key": key, "observations": int(use.sum()),
                   "correction_mean_error_m": float(error[use].mean()),
                   "correction_rmse_m": float(np.sqrt(np.mean(error[use] ** 2)))}
            for name, (nll, nis) in score_cache.items():
                row[f"{name}_mean_nll"] = float(nll[use].mean())
                row[f"{name}_mean_nis"] = float(nis[use].mean())
            writer.writerow(row)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    correction_error = np.linalg.norm(data["residual_world_m"][dev], axis=1)
    x = np.sort(correction_error); axes[0].plot(x * 100, np.arange(1, len(x) + 1) / len(x))
    axes[0].set(xlabel="Corrected position error (cm)", ylabel="Empirical CDF", xlim=(0, np.quantile(x, .99) * 100))
    for name, (residual, cov) in covariance_by_name.items():
        nis = np.sort(score_arrays(residual, cov)[1])
        axes[1].plot(nis, np.arange(1, len(nis) + 1) / len(nis), label=name.split("_")[0])
    axes[1].axvline(5.991, color="black", linestyle="--", linewidth=.8)
    axes[1].set(xlabel="NIS", ylabel="Empirical CDF", xlim=(0, 15)); axes[1].legend(frameon=False)
    fig.tight_layout(); figure_path = staging / "ddev_error_and_calibration.png"
    fig.savefig(figure_path, dpi=180); plt.close(fig)

    report = {
        "schema": "thesis_reference_ddev_evaluation.v1",
        "status": "complete_frozen_working_set_evaluation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "fit_role": "D_R", "evaluation_role": "D_dev", "final_audit_accessed": False,
        "Rproj": {
            "role": "external uncertainty-aware replay/evaluation baseline only",
            "navigation_arm": False, "planning_field": False,
            "sigma_px": sigma_px,
            "fit": "equal aggregate weight per D_R physical position; 2D Gaussian MLE",
            "propagation": "sigma_px^2 I2 through numerical pixel-to-ground Jacobian at box bottom centre",
        },
        "D_dev_covariance_metrics": scores,
        "D_dev_correction_metrics": correction_manifest["D_dev_metrics"],
        "sources": {
            "correction_manifest": {"path": str((correction / "manifest.json").relative_to(REPO)),
                                    "sha256": sha256(correction / "manifest.json")},
            "covariance_manifest": {"path": str((covariance / "manifest.json").relative_to(REPO)),
                                    "sha256": sha256(covariance / "manifest.json")},
        },
        "artifacts": {
            "per_position": {"path": per_position_path.name, "sha256": sha256(per_position_path)},
            "figure": {"path": figure_path.name, "sha256": sha256(figure_path)},
        },
        "implementation": str(Path(__file__).resolve().relative_to(REPO)),
        "implementation_sha256": sha256(Path(__file__).resolve()),
    }
    manifest_path = staging / "manifest.json"
    manifest_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (staging / ".complete").write_text(json.dumps({"manifest_sha256": sha256(manifest_path)}) + "\n")
    os.replace(staging, output)
    print(json.dumps({"status": report["status"], "sigma_px": sigma_px,
                      "D_dev_covariance_metrics": scores}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
