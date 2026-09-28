#!/usr/bin/env python3
"""Analyse the camera-removal campaign: one row per run, per-arm tables, matched differences.

Reads logs/thesis/final_campaign/campaign/seed*/campaign_log.json (the runner's canonical
ledger) and each run's own artifacts; writes logs/thesis/final_campaign/analysis/:
  runs.csv          one row per campaign cell (5 tasks x 6 conditions x 3 seeds)
  collisions.json   the offline footprint score of every run (pipeline/score_collisions.py)
  summary.json      per-arm counts and means, and matched differences with bootstrap CIs
  trajectories.png  true paths per task and model, intact solid and removal dashed

Metrics are those of METHOD §11. Success (amendment D): the run stopped at the goal on the
belief, its true final goal distance is below 0.30 m, it never left the driveable region,
and its evidence is complete. A cell whose run stayed infra_invalid after the retry has no
outcome; it is reported as missing and drops out of every matched pair that needs it.
Matched differences pair runs on (task, seed); CIs are 95 % percentile bootstraps over
those pairs.

    python3 pipeline/analyze_campaign.py
"""
from __future__ import annotations

import csv
import concurrent.futures
import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src/unav_common")]
from pipeline.score_collisions import DriveableRegion, read_poses, score_poses  # noqa: E402

FINAL_ROOT = REPO / "logs/thesis/final_campaign"
CAMPAIGN = FINAL_ROOT / "campaign"
ROUTES = FINAL_ROOT / "routes"
CONFIGS = FINAL_ROOT / "campaign_configs"
OUT = FINAL_ROOT / "analysis"
SEEDS = (91500, 91501, 91502)
MODELS = ("global", "per_camera", "spatial")
STATES = ("intact", "removal")
SUCCESS_GOAL_DISTANCE_M = 0.30
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20260924
METRICS = ("success", "final_goal_distance_m", "belief_error_m", "belief_sigma_major_m",
           "path_length_m", "duration_s", "min_clearance_m")


def tasks() -> list[str]:
    cfg = yaml.safe_load((CONFIGS / "campaign_seed91500.yaml").read_text())
    return list(cfg["tasks"])


def split_condition(condition: str) -> tuple[str, str]:
    model, state = condition.rsplit("_", 1)
    return model, state


def mean_after(path: Path, column: str, start_s: float) -> float:
    values = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                stamp, value = float(row["stamp"]), float(row[column])
            except (KeyError, TypeError, ValueError):
                continue
            if stamp >= start_s and math.isfinite(value):
                values.append(value)
    return float(np.mean(values)) if values else math.nan


def route_name(task: str, condition: str, route_sha: str, routes: Path = ROUTES) -> str:
    result = json.loads((routes / task / "manifest.json").read_text())["results"][condition]
    if result["preselected_route"]["sha256"] != route_sha:
        raise RuntimeError(f"{task}/{condition}: run used a route other than the solved one")
    return result["selected_source"].split(":")[-1]


def score_run(run_dir: str) -> tuple[str, dict]:
    """Score one independent run in a worker process."""
    run = Path(run_dir)
    region = DriveableRegion.for_world()
    return run_dir, score_poses(region, read_poses(run / "ground_truth_pose.csv"))


def run_row(task: str, condition: str, seed: int, entry: dict | None,
            score: dict | None) -> tuple[dict, dict]:
    model, state = split_condition(condition)
    row = {"task": task, "condition": condition, "model": model, "state": state, "seed": seed}
    if entry is None or entry.get("outcome") in (None, "infra_invalid"):
        row["outcome"] = "missing" if entry is None else "infra_invalid"
        return row, {}
    required_entry_flags = (
        "attempt_evidence_complete", "route_artifact_verified",
        "correction_assimilation_verified", "detector_journal_verified",
        "manager_journal_verified",
    )
    failed_flags = [name for name in required_entry_flags if entry.get(name) is not True]
    if failed_flags or entry.get("process_returncode") != 0:
        raise RuntimeError(
            f"{task}/{condition}/seed{seed}: invalid campaign evidence: "
            f"flags={failed_flags}, returncode={entry.get('process_returncode')}"
        )
    run = Path(entry["run_dir"])
    summary = json.loads((run / "run_summary.json").read_text())
    if (summary.get("valid_run") is not True
            or summary.get("evidence_complete") is not True
            or summary.get("runtime_event_invalid_count") != 0
            or summary.get("correction_ledger", {}).get("valid") is not True):
        raise RuntimeError(f"{run}: run summary failed the final evidence audit")
    if score is None:
        raise RuntimeError(f"{run}: missing offline collision score")
    first_cmd = float(summary["first_cmd_stamp"])
    collision = bool(score["collision"])
    goal_distance = float(summary["final_goal_distance"])
    if summary.get("final_goal_distance_reference") != "ground_truth":
        raise RuntimeError(f"{run}: final goal distance is not ground truth")
    stopped = entry["outcome"] == "goal_reached"
    success = stopped and not collision and goal_distance < SUCCESS_GOAL_DISTANCE_M
    row.update({
        "outcome": entry["outcome"],
        "completion_reason": entry["completion_reason"],
        "route": route_name(task, condition, entry["preselected_route_sha256"], ROUTES),
        "success": int(success),
        "failure": ("" if success else "collision" if collision else
                    "goal_distance" if stopped else entry["outcome"]),
        "collision": int(collision),
        "collision_stamp_s": (score["first_exit"] or {}).get("stamp_s", ""),
        "collision_x": (score["first_exit"] or {}).get("x", ""),
        "collision_y": (score["first_exit"] or {}).get("y", ""),
        "final_goal_distance_m": goal_distance,
        "belief_error_m": float(summary["mean_belief_error_gt_after_first_cmd_m"]),
        "belief_sigma_major_m": mean_after(run / "experiment.csv", "state_sigma_major_m", first_cmd),
        "path_length_m": float(summary["path_length_m"]),
        "duration_s": float(summary["stop_stamp"]) - first_cmd,
        "min_clearance_m": min(score["min_obstacle_clearance_m"], score["min_boundary_clearance_m"]),
        "run_dir": str(run.relative_to(REPO)),
    })
    return row, score


def bootstrap_ci(values, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"n": 0, "mean": None, "ci95": None}
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, size=(resamples, values.size))].mean(axis=1)
    return {"n": int(values.size), "mean": float(values.mean()),
            "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]}


def valid(row: dict) -> bool:
    return row.get("outcome") not in ("missing", "infra_invalid")


def matched_differences(rows: list[dict], treatment: str, baseline: str) -> dict:
    """treatment minus baseline, paired on (task, seed), for every metric."""
    by_cell = {(r["task"], r["seed"], r["condition"]): r for r in rows if valid(r)}
    pairs = [(by_cell[(t, s, treatment)], by_cell[(t, s, baseline)])
             for (t, s, c) in by_cell if c == treatment and (t, s, baseline) in by_cell]
    out = {}
    for metric in METRICS:
        diffs = [a[metric] - b[metric] for a, b in pairs
                 if math.isfinite(a[metric]) and math.isfinite(b[metric])]
        out[metric] = bootstrap_ci(diffs)
    return out


def arm_table(rows: list[dict]) -> dict:
    table = {}
    for model in MODELS:
        for state in STATES:
            arm = [r for r in rows if r["model"] == model and r["state"] == state]
            ok = [r for r in arm if valid(r)]
            outcomes = {}
            for r in arm:
                outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
            entry = {"cells": len(arm), "valid": len(ok), "outcomes": outcomes,
                     "successes": sum(r["success"] for r in ok),
                     "collisions": sum(r["collision"] for r in ok)}
            for metric in METRICS[1:]:
                values = [r[metric] for r in ok if math.isfinite(r[metric])]
                entry[metric] = {"mean": float(np.mean(values)) if values else None,
                                 "median": float(np.median(values)) if values else None}
            table[f"{model}_{state}"] = entry
    return table


def plot_trajectories(rows: list[dict], task_names: list[str]) -> None:
    from unav_common.occlusion_geometry import profile_collision_scene
    world = REPO / "src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf"
    profile = yaml.safe_load((REPO / "src/experiments/config/world_profiles.yaml").read_text())["worlds"][world.name]
    prisms = profile_collision_scene(str(world), profile).prisms
    site = next(r for r in profile["known_2d_regions"] if r.get("type") == "site_boundary")
    colours = {"global": "#657789", "per_camera": "#c57a2a", "spatial": "#3f7f5f"}
    fig, axes = plt.subplots(len(task_names), 3, figsize=(15, 4.3 * len(task_names)), squeeze=False)
    for i, task in enumerate(task_names):
        for j, model in enumerate(MODELS):
            ax = axes[i, j]
            ax.add_patch(plt.Rectangle((site["xmin"], site["ymin"]), site["xmax"] - site["xmin"],
                                       site["ymax"] - site["ymin"], fill=False, ec="#3b6fb6", ls="--"))
            for p in prisms:
                ax.add_patch(plt.Rectangle((p.xmin, p.ymin), p.xmax - p.xmin, p.ymax - p.ymin,
                                           fc="#d9d6cc", ec="#9a978f", lw=0.4))
            for r in rows:
                if r["task"] != task or r["model"] != model or not valid(r):
                    continue
                poses = np.array([p[1:3] for p in read_poses(REPO / r["run_dir"] / "ground_truth_pose.csv")])
                ax.plot(poses[:, 0], poses[:, 1], "-" if r["state"] == "intact" else "--",
                        color=colours[model], lw=1.2, alpha=0.8)
                if r["collision"]:  # where the footprint first left the driveable region
                    ax.plot(r["collision_x"], r["collision_y"], "x", color="red", ms=10, mew=2, zorder=5)
            cells = [r for r in rows if r["task"] == task and r["model"] == model]
            wins = {s: sum(r.get("success", 0) for r in cells if r["state"] == s) for s in STATES}
            ax.set_title(f"{task.replace('thesis10_', '')}\n{model}: success intact {wins['intact']}/3, "
                         f"removal {wins['removal']}/3", fontsize=9)
            ax.set_aspect("equal"); ax.set_xlim(-12, 12); ax.set_ylim(-10, 10)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Campaign true paths: intact (solid), camera removed (dashed), 3 seeds each; "
                 "x = first exit from the driveable region", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(OUT / "trajectories.png", dpi=110)
    plt.close(fig)


def source_identities() -> dict:
    """The commit and config each log root's runner froze, per seed."""
    out = {}
    for seed in SEEDS:
        path = CAMPAIGN / f"seed{seed}/source_snapshot/source_identity.json"
        if path.is_file():
            identity = json.loads(path.read_text())
            out[str(path.parent.parent.relative_to(REPO))] = {
                "git_sha": identity["git_provenance"]["git_sha"],
                "git_dirty": identity["git_provenance"]["git_dirty"],
                "campaign_config_sha256": identity["campaign_config_sha256"]}
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    task_names = tasks()
    selected = []
    for seed in SEEDS:
        path = CAMPAIGN / f"seed{seed}/campaign_log.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing final campaign ledger: {path}")
        ledger = json.loads(path.read_text())
        for task in task_names:
            for model in MODELS:
                for state in STATES:
                    condition = f"{model}_{state}"
                    entry = ledger.get(f"{task}__{condition}__seed{seed}")
                    selected.append((task, condition, seed, entry))

    run_dirs = [str(Path(entry["run_dir"])) for _, _, _, entry in selected
                if entry is not None and entry.get("outcome") not in (None, "infra_invalid")]
    workers = min(8, os.cpu_count() or 1, len(run_dirs))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        scored = dict(executor.map(score_run, run_dirs))

    rows, scores = [], {}
    for task, condition, seed, entry in selected:
        score = scored.get(str(Path(entry["run_dir"]))) if entry else None
        row, score = run_row(task, condition, seed, entry, score)
        rows.append(row)
        if score:
            scores[row["run_dir"]] = score
    fields = ["task", "condition", "model", "state", "seed", "outcome", "completion_reason", "route",
              "success", "failure", "collision", "collision_stamp_s", "collision_x", "collision_y",
              *METRICS[1:], "run_dir"]
    with (OUT / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "collisions.json").write_text(json.dumps(scores, indent=1, default=str) + "\n")
    summary = {
        "schema": "thesis_campaign_analysis.v1",
        "source_identity": source_identities(),
        "success_rule": f"stopped at goal on belief, true final goal distance < {SUCCESS_GOAL_DISTANCE_M} m, "
                        "no footprint exit, evidence complete",
        "cells": len(rows), "valid": sum(valid(r) for r in rows),
        "arms": arm_table(rows),
        "removal_minus_intact": {m: matched_differences(rows, f"{m}_removal", f"{m}_intact") for m in MODELS},
        "spatial_minus_other_under_removal": {
            m: matched_differences(rows, "spatial_removal", f"{m}_removal") for m in ("global", "per_camera")},
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED, "unit": "(task, seed) pair"},
    }
    if summary["cells"] != 90 or summary["valid"] != 90:
        raise RuntimeError(
            f"final campaign is incomplete: cells={summary['cells']}, valid={summary['valid']}"
        )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    plot_trajectories(rows, task_names)
    print(json.dumps({"out": str(OUT), "cells": summary["cells"], "valid": summary["valid"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
