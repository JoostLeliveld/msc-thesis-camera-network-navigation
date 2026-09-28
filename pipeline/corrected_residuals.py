#!/usr/bin/env python3
"""Corrected residuals of the frozen correction, in the world and camera-ray frames.

Input to pipeline/fit_covariance.py (the one R0/R1/R2 family) and to
pipeline/evaluate_ddev.py. Fits nothing and selects nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src/reliability"), str(REPO / "src/unav_common"), str(REPO)]
from pipeline.dataset import load_rows  # noqa: E402
from reliability.projection import camera_model_from_world  # noqa: E402

CAMERAS = tuple(f"camera_{letter}" for letter in "ABCDE")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    correction, output = args.correction.resolve(), args.output.resolve()
    staging = output.with_name(output.name + ".incomplete")
    if output.exists() or staging.exists(): raise FileExistsError(output if output.exists() else staging)
    staging.mkdir(parents=True)
    correction_manifest_path = correction / "manifest.json"
    correction_manifest = json.loads(correction_manifest_path.read_text(encoding="utf-8"))
    if correction_manifest.get("status") != "frozen_before_covariance_fit":
        raise RuntimeError("correction is not frozen")
    prediction_path = correction / correction_manifest["artifacts"]["predictions"]["path"]
    if sha256(prediction_path) != correction_manifest["artifacts"]["predictions"]["sha256"]:
        raise RuntimeError("correction predictions hash drift")
    with np.load(prediction_path, allow_pickle=False) as source:
        data = {name: np.asarray(source[name]) for name in source.files}
    rows = load_rows()
    world = REPO / "src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf"
    models = {c: camera_model_from_world(world, include_name=next(
        r["camera_model"] for r in rows if r["camera_id"] == c)) for c in CAMERAS}
    camera_xy = {c: np.asarray(models[c].cam_pos[:2], dtype=float) for c in CAMERAS}
    residual_world = data["residual_world_m"].astype(float)
    corrected = data["corrected_xy_m"].astype(float)
    residual_ray = np.empty_like(residual_world)
    for i, (camera_id, point, residual) in enumerate(zip(data["camera"], corrected, residual_world)):
        along = point - camera_xy[str(camera_id)]; along /= np.linalg.norm(along)
        basis = np.column_stack((along, np.asarray([-along[1], along[0]])))
        residual_ray[i] = basis.T @ residual
    residual_path = staging / "corrected_residuals.npz"
    np.savez_compressed(
        residual_path, role=data["role"], position_key=data["position_key"], camera=data["camera"],
        truth_xy_m=data["truth_xy_m"], corrected_xy_m=corrected,
        residual_world_m=residual_world, residual_ray_m=residual_ray,
    )
    report = {
        "schema": "thesis_corrected_residuals.v1", "status": "frozen_before_covariance_fit",
        "created_utc": datetime.now(timezone.utc).isoformat(), "final_audit_accessed": False,
        "roles": sorted(set(data["role"].astype(str).tolist())),
        "correction_manifest": str(correction_manifest_path.relative_to(REPO)),
        "correction_manifest_sha256": sha256(correction_manifest_path),
        "artifacts": {"residuals": {"path": residual_path.name, "sha256": sha256(residual_path)}},
        "implementation": str(Path(__file__).resolve().relative_to(REPO)),
        "implementation_sha256": sha256(Path(__file__).resolve()),
    }
    report_path = staging / "manifest.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (staging / ".complete").write_text(json.dumps({"manifest_sha256": sha256(report_path)}) + "\n")
    os.replace(staging, output)
    print(json.dumps({"status": report["status"], "residuals": str(output / residual_path.name)}))
    return 0


if __name__ == "__main__": raise SystemExit(main())
