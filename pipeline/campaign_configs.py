#!/usr/bin/env python3
"""Bind the campaign templates to the current fits: route-solving and per-seed execution configs.

The templates hold every navigation, planner and follower setting and repo-relative
artifact paths. This resolves those paths (they must lie under logs/thesis/fits), records
their hashes, and writes logs/thesis/campaign_configs/: the route-solving config and one
execution config per seed, so the campaign can run seed by seed.

    python3 pipeline/campaign_configs.py
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
HERE = REPO / "pipeline"
ROOT = REPO / "logs/thesis"
SEEDS = (91500, 91501, 91502)
GATE = "config/sensor_gate.yaml"
HASH_OF = {"manager_visibility_sensor_model_path": "manager_visibility_sensor_model_expected_sha256",
           "camera_network_artifact_path": "camera_network_expected_sha256"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bind(cfg: dict) -> dict:
    """Resolve the template's repo-relative artifact paths and record their hashes."""
    if cfg["manager_sensor_gate_config_path"] != GATE:
        raise RuntimeError("template does not use the fitted sensor gate")
    detector = REPO / cfg["yolo_model"]
    if not detector.is_file():
        raise FileNotFoundError(detector)
    for condition in cfg["conditions"].values():
        for key, hash_key in HASH_OF.items():
            if key in condition:
                path = (REPO / condition[key]).resolve(strict=True)
                if not str(path).startswith(str(ROOT / "fits")):
                    raise RuntimeError(f"{key} is not a current fit: {path}")
                condition[key] = str(path)
                condition[hash_key] = sha(path)
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "campaign_configs",
                        help="Directory for generated planning and per-seed configs.")
    parser.add_argument("--only-task", action="append", default=[],
                        help="Include only this task; repeat to create a scoped campaign revision.")
    args = parser.parse_args()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    planning = bind(yaml.safe_load((HERE / "route_planning_template.yaml").read_text()))
    if args.only_task:
        requested = set(args.only_task)
        unknown = requested.difference(planning["tasks"])
        if unknown:
            raise ValueError("unknown --only-task value(s): " + ", ".join(sorted(unknown)))
        planning["tasks"] = {name: cfg for name, cfg in planning["tasks"].items() if name in requested}
    (out_dir / "route_planning_campaign.yaml").write_text(yaml.safe_dump(planning, sort_keys=False))
    execution = bind(yaml.safe_load((HERE / "execution_template.yaml").read_text()))
    if args.only_task:
        execution["tasks"] = {name: cfg for name, cfg in execution["tasks"].items()
                              if name in requested}
    # Task-visibility rule: every start and goal must stay seen with the task's camera dropped.
    import sys
    sys.path.insert(0, str(REPO))
    from pipeline.check_task_visibility import failures
    failed = failures(execution)
    if failed:
        raise SystemExit("task-visibility rule failed:\n  " + "\n  ".join(failed))
    for seed in SEEDS:
        per_seed = yaml.safe_load(yaml.safe_dump(execution))
        per_seed["study_title"] = f"camera-removal campaign, seed {seed}"
        for task in per_seed["tasks"].values():
            task["seeds"] = [seed]
        (out_dir / f"execution_template_seed{seed}.yaml").write_text(
            yaml.safe_dump(per_seed, sort_keys=False))
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
