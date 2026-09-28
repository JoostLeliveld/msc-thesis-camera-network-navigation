#!/usr/bin/env python3
"""Fit and freeze the canonical D_mu structured-plus-visibility correction."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src/reliability"), str(REPO / "src/unav_common")]
from reliability.visibility_residual_net import VisibilityPatchResidualNet  # noqa: E402

SEED = 260921
EPOCHS = 80
BATCH_SIZE = 128
FEATURE_NAMES = (
    "raw_range_m", "inverse_raw_range", "ray_bearing_sin", "ray_bearing_cos",
    "bbox_width_fraction", "bbox_height_fraction", "bbox_aspect",
    "bbox_bottom_u_fraction", "bbox_bottom_v_fraction", "confidence",
    *(f"is_camera_{letter}" for letter in "ABCDE"),
)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def position_weights(position: np.ndarray, index: np.ndarray) -> np.ndarray:
    counts = Counter(position[index].tolist())
    weight = np.asarray([1.0 / counts[value] for value in position[index]], dtype=np.float64)
    return weight / weight.mean()


def position_metrics(target: np.ndarray, prediction: np.ndarray, position: np.ndarray,
                     camera: np.ndarray, index: np.ndarray) -> dict:
    residual = prediction[index] - target[index]
    error = np.linalg.norm(residual, axis=1)
    per_position = []
    for value in sorted(set(position[index].tolist())):
        use = position[index] == value
        per_position.append({
            "position_key": value, "n": int(use.sum()),
            "mean_error_m": float(error[use].mean()),
            "rmse_m": float(np.sqrt(np.mean(error[use] ** 2))),
            "signed_bias_ray_m": residual[use].mean(axis=0).tolist(),
        })
    result = {
        "observations": int(len(index)), "positions": len(per_position),
        "equal_position_mean_error_m": float(np.mean([x["mean_error_m"] for x in per_position])),
        "equal_position_rmse_m": float(np.mean([x["rmse_m"] for x in per_position])),
        "equal_position_signed_bias_ray_m": np.mean(
            [x["signed_bias_ray_m"] for x in per_position], axis=0
        ).tolist(),
        "pooled_median_m": float(np.median(error)),
        "pooled_p95_m": float(np.quantile(error, 0.95)),
        "by_camera": {},
    }
    for value in sorted(set(camera[index].tolist())):
        members = index[camera[index] == value]
        local = np.linalg.norm(prediction[members] - target[members], axis=1)
        result["by_camera"][value] = {
            "n": int(len(members)), "median_m": float(np.median(local)),
            "rmse_m": float(np.sqrt(np.mean(local ** 2))),
            "p95_m": float(np.quantile(local, 0.95)),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.gate_dataset.resolve(), args.output.resolve()
    staging = output.with_name(output.name + ".incomplete")
    if output.exists() or staging.exists():
        raise FileExistsError(output if output.exists() else staging)
    staging.mkdir(parents=True)
    source_manifest_path = source / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    tensor_path = source / source_manifest["admitted_tensors"]
    if source_manifest.get("status") != "complete" or source_manifest.get("final_audit_accessed"):
        raise RuntimeError("gate dataset is not a sealed working-role artifact")
    if sha256(tensor_path) != source_manifest["admitted_tensors_sha256"]:
        raise RuntimeError("gate tensor hash drift")

    with np.load(tensor_path, allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}
    if tuple(data["feature_names"].tolist()) != FEATURE_NAMES:
        raise RuntimeError("structured feature contract drift")
    role, position, camera = data["role"], data["position_key"], data["camera"]
    feature = data["structured_feature"].astype(np.float64)
    grid = data["visibility_grid"].astype(np.float32)
    target = data["target_ray_m"].astype(np.float64)
    train = np.flatnonzero(role == "D_mu")
    development = np.flatnonzero(role == "D_dev")
    covariance = np.flatnonzero(role == "D_R")
    role_positions = {
        name: set(position[role == name].tolist()) for name in ("D_mu", "D_R", "D_dev")
    }
    if (role_positions["D_mu"] & role_positions["D_R"]
            or role_positions["D_mu"] & role_positions["D_dev"]
            or role_positions["D_R"] & role_positions["D_dev"]):
        raise RuntimeError("admitted positions cross canonical role boundaries")
    sample_weight = position_weights(position, train)

    # CPU only: seeded CUDA training is not deterministic (two runs on identical inputs
    # gave different visibility weights and flipped the R2 neighbour count), while two
    # CPU runs are byte-identical.
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cpu")
    base = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(96, 64), activation="relu", solver="adam",
            alpha=1e-4, batch_size=128, learning_rate_init=1e-3,
            max_iter=500, early_stopping=False, random_state=SEED,
        ),
    )
    base.fit(feature[train], target[train], mlpregressor__sample_weight=sample_weight)
    base_prediction = base.predict(feature)
    base_path = staging / "box_mlp.joblib"
    joblib.dump(base, base_path)

    feature_mean = np.average(feature[train], axis=0, weights=sample_weight)
    centered = feature[train] - feature_mean
    feature_sd = np.sqrt(np.average(centered ** 2, axis=0, weights=sample_weight))
    feature_sd = np.maximum(feature_sd, 1e-3)
    model = VisibilityPatchResidualNet(feature.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    tail = np.linalg.norm(target[train] - base_prediction[train], axis=1)
    edges = np.unique(np.quantile(tail, (0.50, 0.80, 0.95)))
    bins = np.searchsorted(edges, tail, side="right")
    bin_counts = np.bincount(bins, minlength=len(edges) + 1).astype(float)
    tail_weight = np.asarray([len(bins) / max(bin_counts[value], 1.0) for value in bins])
    weight = np.clip(sample_weight * tail_weight, 0.1, 12.0).astype(np.float32)
    weight /= weight.mean()
    generator = np.random.default_rng(SEED + 1709)
    history = []
    for epoch in range(EPOCHS):
        model.train(); losses = []
        order = generator.permutation(len(train))
        for start in range(0, len(order), BATCH_SIZE):
            local = order[start:start + BATCH_SIZE]; part = train[local]
            g = torch.from_numpy(grid[part]).to(device)
            f = torch.from_numpy(((feature[part] - feature_mean) / feature_sd).astype(np.float32)).to(device)
            b = torch.from_numpy(base_prediction[part].astype(np.float32)).to(device)
            truth = torch.from_numpy(target[part].astype(np.float32)).to(device)
            w = torch.from_numpy(weight[local]).to(device)
            prediction, residual, gate_value = model(g, f, b)
            loss_axis = F.smooth_l1_loss(prediction, truth, reduction="none", beta=0.02)
            loss = (loss_axis.mean(dim=1) * w).mean() + 2e-4 * (gate_value * residual).square().mean()
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step(); history.append(float(np.mean(losses)))
        if (epoch + 1) % 10 == 0:
            print(f"visibility epoch {epoch + 1}/{EPOCHS} loss={history[-1]:.6f}", flush=True)

    prediction = np.empty_like(target)
    gate_values = np.empty(len(target), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(target), 512):
            part = np.arange(start, min(start + 512, len(target)))
            result, _, gate_value = model(
                torch.from_numpy(grid[part]).to(device),
                torch.from_numpy(((feature[part] - feature_mean) / feature_sd).astype(np.float32)).to(device),
                torch.from_numpy(base_prediction[part].astype(np.float32)).to(device),
            )
            prediction[part] = result.cpu().numpy(); gate_values[part] = gate_value[:, 0].cpu().numpy()
    patch_path = staging / "visibility_residual.pt"
    torch.save({
        "schema": "visibility_patch_residual.v1", "grid_size": 16,
        "feature_names": FEATURE_NAMES, "state_dict": model.cpu().state_dict(),
        "feature_mean": feature_mean.tolist(), "feature_sd": feature_sd.tolist(),
    }, patch_path)
    prediction_path = staging / "predictions.npz"
    corrected = data["raw_xy_m"] + np.einsum("nij,nj->ni", data["ray_basis"], prediction)
    residual_world = corrected - data["truth_xy_m"]
    covariance_basis = np.empty_like(data["ray_basis"])
    # Camera coordinates are not needed here: rotate world residual through the
    # correction basis now; Stage 07 recomputes the corrected-observation basis.
    np.savez_compressed(
        prediction_path, role=role, position_key=position, camera=camera,
        plan_pose_index=data["plan_pose_index"], yaw_idx=data["yaw_idx"],
        truth_xy_m=data["truth_xy_m"], raw_xy_m=data["raw_xy_m"],
        correction_ray_m=prediction, corrected_xy_m=corrected,
        residual_world_m=residual_world, base_prediction_ray_m=base_prediction,
        visibility_gate=gate_values, bbox_xyxy=data["bbox_xyxy"],
        confidence=data["confidence"], image_sha1=data["image_sha1"],
    )
    metrics = {
        "raw": position_metrics(target, np.zeros_like(target), position, camera, development),
        "structured_only": position_metrics(target, base_prediction, position, camera, development),
        "structured_plus_visibility": position_metrics(target, prediction, position, camera, development),
    }
    report = {
        "schema": "thesis_reference_correction.v1", "status": "frozen_before_covariance_fit",
        "created_utc": datetime.now(timezone.utc).isoformat(), "final_audit_accessed": False,
        "fit_role": "D_mu", "development_role": "D_dev", "covariance_role_untouched": "D_R",
        "fit_observations": int(len(train)), "fit_positions": len(role_positions["D_mu"]),
        "development_observations": int(len(development)),
        "development_positions": len(role_positions["D_dev"]),
        "covariance_support_observations": int(len(covariance)),
        "covariance_support_positions": len(role_positions["D_R"]),
        "position_weighting": "each physical position has equal aggregate training weight",
        "selected_model": "structured_plus_16x16_visibility_residual",
        "selection_status": "predeclared deployed model; D_dev reports diagnostics but does not reopen model family",
        "feature_names": list(FEATURE_NAMES),
        "architecture": {
            "structured_base": "StandardScaler + MLPRegressor(96,64), ReLU",
            "visibility_branch": "Conv(1,8)-Conv(8,16,stride2)-Conv(16,24,stride2)-pool4x4",
            "fusion": "structured32 + visibility384 -> 96 -> 48 -> gated 2D residual",
            "loss": "position-balanced tail-balanced SmoothL1 beta=0.02 + gated-residual L2",
        },
        "training": {"seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
                     "optimizer": "AdamW(lr=8e-4,weight_decay=1e-4)+cosine", "loss_history": history},
        "D_dev_metrics": metrics,
        "gate_manifest": str(source_manifest_path.relative_to(REPO)),
        "gate_manifest_sha256": sha256(source_manifest_path),
        "artifacts": {
            "base": {"path": base_path.name, "sha256": sha256(base_path)},
            "patch": {"path": patch_path.name, "sha256": sha256(patch_path)},
            "predictions": {"path": prediction_path.name, "sha256": sha256(prediction_path)},
        },
        "implementation": str(Path(__file__).resolve().relative_to(REPO)),
        "implementation_sha256": sha256(Path(__file__).resolve()),
    }
    report_path = staging / "manifest.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (staging / ".complete").write_text(json.dumps({"manifest_sha256": sha256(report_path)}) + "\n")
    os.replace(staging, output)
    print(json.dumps({"status": report["status"], "D_dev_metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
