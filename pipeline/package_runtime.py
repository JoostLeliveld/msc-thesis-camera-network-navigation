#!/usr/bin/env python3
"""Package the frozen correction with canonical ray-frame R0--R2 artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


METHODS = (
    "R0_global_full", "R1_per_camera_full", "R2_spatial_residual",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified(path: Path) -> dict[str, str]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": digest(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correction-manifest", required=True, type=Path)
    parser.add_argument("--covariance-models", required=True, type=Path)
    parser.add_argument("--world", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    correction_path = args.correction_manifest.resolve()
    covariance_path = args.covariance_models.resolve()
    correction = json.loads(correction_path.read_text(encoding="utf-8"))
    if correction.get("status") != "frozen_before_covariance_fit":
        raise ValueError("correction manifest is not frozen")
    if correction.get("selected_model") != "structured_plus_16x16_visibility_residual":
        raise ValueError("runtime package requires the selected visibility correction")
    correction_root = correction_path.parent
    base = correction_root / correction["artifacts"]["base"]["path"]
    patch = correction_root / correction["artifacts"]["patch"]["path"]
    if digest(base) != correction["artifacts"]["base"]["sha256"]:
        raise ValueError("correction base hash drift")
    if digest(patch) != correction["artifacts"]["patch"]["sha256"]:
        raise ValueError("correction patch hash drift")
    world = args.world.resolve()
    import sys
    repo = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repo / "src/reliability"), str(repo / "src/unav_common")]
    from reliability.projection import camera_model_from_world
    camera_order = [f"camera_{letter}" for letter in "ABCDE"]
    includes = [f"external_camera{suffix}" for suffix in ("", "_b", "_c", "_d", "_e")]
    camera_xy = {camera: list(map(float, camera_model_from_world(
        world, include_name=include).cam_pos[:2])) for camera, include in zip(camera_order, includes)}

    output = args.output.resolve()
    staging = output.with_name(output.name + ".incomplete")
    if output.exists() or staging.exists():
        raise FileExistsError(output if output.exists() else staging)
    staging.mkdir(parents=True)
    covariance_entry = verified(covariance_path)
    artifacts = {}
    for method in METHODS:
        manifest = {
            "schema": "commissioned_visibility_sensor_model.v2",
            "status": "frozen_before_audit",
            "audit_accessed": False,
            "mean_model": "box_mlp_visibility_residual",
            "runtime_covariance_model": method,
            "planner_covariance_model": (
                "not_used_replay_only" if method == "Rproj_homography_pixel" else method),
            "covariance_parameterization": "camera_to_query_ray_frame",
            "covariance_runtime_frame": "map_bev_after_query_dependent_rotation",
            "covariance_statistic": "position-balanced second moment about zero",
            "camera_order": camera_order,
            "camera_xy_m": camera_xy,
            "image_shape_hw": [720, 1280],
            "visibility_grid": correction.get(
                "visibility_grid", {"shape": [1, 16, 16]}),
            "correction_base": verified(base),
            "correction_patch": verified(patch),
            "covariance_models": covariance_entry,
            "correction_manifest": verified(correction_path),
        }
        path = staging / f"{method}.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
        artifacts[method] = {
            "path": str((output / path.name).resolve()),
            "sha256": digest(path),
        }
    index = {
        "schema": "canonical_r012_runtime_manifest.v1",
        "correction_manifest": verified(correction_path),
        "covariance_models": covariance_entry,
        "artifacts": artifacts,
    }
    (staging / "manifest.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(staging, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
