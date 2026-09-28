#!/usr/bin/env python3
"""Bind complete offline Stage-10 EFE solves into a preselected-route campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import yaml

import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src/experiments"), str(REPO / "src/unav_common")]
from experiments.core.world_profiles import serialize_driveable_geometry_from_profile  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--routes-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.campaign.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    profile_path = (REPO / config["world_profiles"]).resolve()
    profiles = yaml.safe_load(profile_path.read_text(encoding="utf-8"))["worlds"]
    config["driveable_geometry_json"] = serialize_driveable_geometry_from_profile(
        profiles[config["world"]]
    )
    for task_name, task_cfg in config["tasks"].items():
        task_dir = (args.routes_root / task_name).resolve()
        manifest_path = task_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "frozen_before_gazebo_execution":
            raise RuntimeError(f"incomplete route solve for {task_name}")
        routes = {}
        for condition in task_cfg["conditions"]:
            result = manifest["results"][condition]
            route_path = task_dir / result["preselected_route"]["path"]
            source_path = task_dir / result["artifact"]["path"]
            route_json = route_path.read_text(encoding="utf-8")
            if sha(source_path) != result["preselected_route"]["source_sha256"]:
                raise RuntimeError(f"source hash drift for {task_name}/{condition}")
            routes[condition] = {
                "preselected_route_json": route_json,
                "preselected_route_sha256": result["preselected_route"]["sha256"],
                "preselected_route_source_path": str(source_path.relative_to(REPO)),
                "preselected_route_source_sha256": sha(source_path),
            }
        task_cfg["preselected_routes"] = routes
    # A preselected route is executed, not solved, so no condition may carry a planner field
    # (the runner refuses one) and the objective must not ask for one (the launcher refuses
    # metric_expected_belief without it). Each arm still differs where the method says it
    # should: in its runtime covariance, and in its route, solved offline with its own field.
    for condition_cfg in config["conditions"].values():
        condition_cfg.pop("camera_network_artifact_path", None)
        condition_cfg.pop("camera_network_expected_sha256", None)
    config["camera_network_objective"] = "legacy_pixel_chart"
    config["global_planner_mode"] = "preselected_route"
    config["global_optimizer_multistart"] = False
    config["optimizer_multistart"] = False
    config["optimizer_multistart_include_direct"] = False
    config["preselected_route_clearance_m"] = 0.0
    config["preselected_route_endpoint_tolerance_m"] = 0.10
    config["preselected_route_sample_step_m"] = 0.04
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(config, sort_keys=False, width=1000), encoding="utf-8")
    os.replace(temporary, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
