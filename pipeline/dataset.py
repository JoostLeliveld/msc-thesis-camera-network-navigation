#!/usr/bin/env python3
"""Load the frozen reference-position dataset used by the final thesis pipeline.

v9 is an INDEX over the capture passes, not a copy. Its rows are

* every ok row of the v5 capture, which ran in two passes over one plan
  (`capture_poses_v5.json`): part 1, and part 2 for the poses part 1 never reached or lost.
  Part 2 was launched with its own pose file, so `pose_id` restarts at 0 there; rows are
  joined back onto the plan as pose_id -> own pose file -> (position_key, yaw_idx);
* the 12 camera-C supplement positions (keys `Qnnnn`);
* the 50 uniform top-up positions (keys `Unnnn`, planned by `capture/plan_topup.py`): every
  1 m cell of the operating domain below 8 positions per m^2, scaled by its footprint-valid
  fraction, is filled to target;
* the repair capture of the 4,182 v5 part-1 poses in `v5_robot_absent_poses.json`: in v5
  session 71daa8ab the robot vanished from the simulator at pose 4214 and every later frame
  is empty background although marked ok. Those rows are dropped and the repaired rows take
  their place under the same plan pose index, position key and role. The repair ran in two
  passes; a later pass overrides an earlier one for every plan pose it contains.

Positions inside objects: a position is dropped when the robot footprint at any of its
poses overlaps an object of the world's full collision scene (the model groups and the
top-level included forklift, pallets, bin and pallet jack). This removes the 12 top-up
positions whose 46 poses were planned without those five objects; no v5, repair or
supplement pose overlaps an object.

* the fresh, spatially balanced final audit of v9 (keys `Vnnnn`, planned by
  `capture/plan_rebalance.py`).

* the lane-grid capture (keys `Lnnnn`, planned by `capture/plan_hard_views.py`);
* the final fill capture (keys `Fnnnn`, planned by `capture/plan_fill_v11.py`).

Roles. `captures/v11/partition_v11.csv`, written by `capture/partition_v11.py`, re-splits
every captured position as one dataset: inside each v9 2 x 2 m stratum all positions are
ordered by sha256(seed, key) regardless of capture, and take the final audit, D_dev and D_R
quotas, with the remaining positions assigned to D_mu. Versioned directory names below are
immutable capture provenance, not competing dataset definitions.

Presence check: a pose where at least two cameras should see the robot but none does is
legitimate at one position (racks can hide all its headings), but such a run spanning three
or more positions within one capture session means the robot was absent. `load_rows`
raises if any such run remains, and if a (plan pose, camera) is captured twice.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CAPTURES = REPO / "logs/thesis/captures"
V5 = CAPTURES / "v5"
V8 = CAPTURES / "v8"
V9 = CAPTURES / "v9"
V10 = CAPTURES / "v10"
V11 = CAPTURES / "v11"
PARTITION = V11 / "partition_v11.csv"

V5_PLAN = V5 / "capture_poses_v5.json"
V5_POSITIONS = V5 / "capture_positions_v5.csv"
V5_PLAN_POSES = 10276
# (name, capture dir, the pose file that capture was launched with)
V5_PASSES = (
    ("part1", V5 / "part1", V5 / "capture_poses_v5.json"),
    ("part2", V5 / "part2", V5 / "capture_poses_v5_part2.json"),
)
# (name, capture dir, pose file, plan_pose_index offset)
EXTENSIONS = (
    ("supplement", V8 / "supplement", V8 / "capture_poses_supplement.json",
     V5_PLAN_POSES),
    ("topup", V8 / "topup", V8 / "capture_poses_v8_topup.json",
     V5_PLAN_POSES + 48),
    ("audit_v9", V9 / "audit", V9 / "capture_poses_v9.json",
     V5_PLAN_POSES + 48 + 200),
    ("lane_v10", V10 / "lane/capture", V10 / "lane/capture_poses_v10_lane_kept.json",
     V5_PLAN_POSES + 48 + 200 + 600),
    ("fill_v11", V11 / "fill/capture", V11 / "fill/capture_poses_v11_fill.json",
     V5_PLAN_POSES + 48 + 200 + 600 + 2226),
)
# repair passes in order; a later pass overrides an earlier one for the plan poses it holds
REPAIRS = (
    ("repair", V8 / "repair", V8 / "capture_poses_repair.json"),
    ("repair2", V8 / "repair2", V8 / "capture_poses_repair2.json"),
)
ABSENT_LIST = V8 / "v5_robot_absent_poses.json"
WORLD = REPO / "src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf"
WORLD_PROFILES = REPO / "src/experiments/config/world_profiles.yaml"
ROBOT_TARGET = REPO / "pipeline/capture/robot_target_manifest.json"
SOURCES = V5_PASSES + tuple((n, d, p) for n, d, p, _ in EXTENSIONS) + REPAIRS

LOCK_PATH = REPO / "pipeline/dataset_lock.json"
LOCK = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
CAMPAIGN_ROOT = REPO / LOCK["campaign_root"]
EXPECTED_WORKING_OPPORTUNITIES = int(LOCK["opportunity_accounting"]["expected_working_opportunities"])
EXPECTED_WORKING_UNIQUE_IMAGES = int(LOCK["opportunity_accounting"]["expected_working_unique_images"])

_PLAN_FIELDS = ("stratum", "kind", "anchor", "block_id", "n_cameras")
_DEFAULTS = {"anchor": -1, "n_cameras": -1}


def _load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _check_pose(name: str, row: dict, pose: dict) -> None:
    """The pose file is the authority for where the robot stood."""
    if (abs(float(row["robot_x"]) - pose["x"]) > 1e-6
            or abs(float(row["robot_y"]) - pose["y"]) > 1e-6):
        raise RuntimeError(f"{name}: row pose_id {row['pose_id']} at "
                           f"({row['robot_x']},{row['robot_y']}) does not match its pose file "
                           f"({pose['x']},{pose['y']})")


def _index_rows(directory: Path, only_ok: bool):
    index = directory / "capture_index.csv"
    if not index.is_file():
        raise FileNotFoundError(index)
    with index.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if only_ok and row.get("capture_status") != "ok":
                continue
            yield row


def _v5_rows(only_ok: bool) -> list[dict]:
    plan = {(p["position_key"], int(p["yaw_idx"])): i for i, p in enumerate(_load_json(V5_PLAN))}
    out = []
    for name, directory, pose_file in V5_PASSES:
        poses = _load_json(pose_file)
        for row in _index_rows(directory, only_ok):
            i = int(row["pose_id"])
            if i >= len(poses):
                raise RuntimeError(f"{name}: pose_id {i} outside its pose file")
            p = poses[i]
            _check_pose(name, row, p)
            key = (p["position_key"], int(p["yaw_idx"]))
            if key not in plan:
                raise RuntimeError(f"{name}: pose {key} is not in the v5 plan")
            row = dict(row)
            row.update({"capture_source": name, "position_key": p["position_key"],
                        "yaw_idx": int(p["yaw_idx"]), "plan_pose_index": plan[key]})
            row.update({f: p[f] for f in _PLAN_FIELDS if f in p})
            out.append(row)
    return out


def _nearest_v5_role() -> Callable[[float, float], str]:
    rows = list(csv.DictReader(V5_POSITIONS.open(encoding="utf-8")))
    xy = np.array([[float(r["x"]), float(r["y"])] for r in rows])
    roles = [r["role"] for r in rows]

    def role(x: float, y: float) -> str:
        return roles[int(np.argmin(np.hypot(xy[:, 0] - x, xy[:, 1] - y)))]
    return role


def _extension_rows(only_ok: bool, v8_only: bool = False) -> list[dict]:
    role_of = _nearest_v5_role()
    out = []
    for name, directory, pose_file, offset in EXTENSIONS:
        if v8_only and name in ("audit_v9", "lane_v10", "fill_v11"):
            continue
        poses = _load_json(pose_file)
        for row in _index_rows(directory, only_ok):
            i = int(row["pose_id"])
            p = poses[i]
            _check_pose(name, row, p)
            role = ("none" if name == "audit_v9" else "unassigned" if name in ("lane_v10", "fill_v11")
                    else role_of(p["x"], p["y"]))
            row = dict(row)
            row.update(_DEFAULTS)
            row.update({"capture_source": name, "position_key": p["position_key"],
                        "yaw_idx": int(p["yaw_idx"]), "plan_pose_index": offset + i,
                        "stratum": role, "kind": p.get("kind", name), "block_id": name})
            out.append(row)
    return out


def _repair_rows(only_ok: bool) -> list[dict]:
    passes = []
    for name, directory, pose_file in REPAIRS:
        poses = _load_json(pose_file)
        rows = []
        if (directory / "capture_index.csv").is_file():
            for row in _index_rows(directory, only_ok):
                p = poses[int(row["pose_id"])]
                _check_pose(name, row, p)
                row = dict(row)
                row.update({"capture_source": name, "position_key": p["position_key"],
                            "yaw_idx": int(p["yaw_idx"]),
                            "plan_pose_index": int(p["v5_plan_pose_index"])})
                row.update({f: p[f] for f in _PLAN_FIELDS if f in p})
                rows.append(row)
        passes.append(({int(p["v5_plan_pose_index"]) for p in poses}, rows))
    out = []
    for i, (_, rows) in enumerate(passes):
        overridden = set().union(*(planned for planned, _ in passes[i + 1:]))
        out += [r for r in rows if int(r["plan_pose_index"]) not in overridden]
    return out


def _presence_rejects(rows: list[dict], run_positions: int = 3, min_in_frame: int = 2) -> set:
    """(capture_source, plan_pose_index) of every pose inside a run of empty poses that
    spans at least `run_positions` distinct positions."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["capture_source"], r["capture_session_id"],
                           int(r["pose_id"]), int(r["plan_pose_index"])), []).append(r)
    rejected, run, last = set(), [], None
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2])):
        rs = groups[key]
        in_frame = sum(r.get("nominal_in_frame") in ("1", "True") for r in rs)
        visible = sum(int(float(r.get("semantic_robot_pixels") or 0)) > 0 for r in rs)
        if (key[0], key[1]) != last:
            run, last = [], (key[0], key[1])
        if in_frame >= min_in_frame and visible == 0:
            run.append((key[0], key[3], rs[0]["position_key"]))
            if len({position for _, _, position in run}) >= run_positions:
                rejected.update((source, pose) for source, pose, _ in run)
        else:
            run = []
    return rejected


def positions_inside_objects(rows: list[dict]) -> set[str]:
    """Position keys with at least one pose whose footprint overlaps a collision object."""
    import sys
    import yaml
    sys.path.insert(0, str(REPO / "src/unav_common"))
    from unav_common.occlusion_geometry import profile_collision_scene
    from unav_common.rectangular_footprint import RectangularFootprint
    profile = yaml.safe_load(WORLD_PROFILES.read_text(encoding="utf-8"))["worlds"][WORLD.name]
    body = _load_json(ROBOT_TARGET)["physical_contract"]
    footprint = RectangularFootprint(profile_collision_scene(str(WORLD), profile).prisms,
                                     length=float(body["body_length_m"]),
                                     width=float(body["body_width_m"]))
    poses = {(r["capture_source"], int(r["plan_pose_index"])):
             (r["position_key"], float(r["robot_x"]), float(r["robot_y"]), float(r["robot_yaw"]))
             for r in rows}
    return {key for key, x, y, yaw in poses.values() if footprint.clearance((x, y, yaw)) < 0.0}


def _check_unique(rows: list[dict]) -> None:
    seen: dict[tuple[int, str], str] = {}
    for r in rows:
        k = (int(r["plan_pose_index"]), r["camera_id"])
        if k in seen:
            raise RuntimeError(f"plan pose {k[0]} camera {k[1]} captured in both "
                               f"{seen[k]} and {r['capture_source']}")
        seen[k] = r["capture_source"]


def load_rows(*, only_ok: bool = True, v8_only: bool = False,
              apply_partition: bool = True) -> list[dict]:
    """Every dataset row, keyed onto the plan, with capture_source, position_key, yaw_idx,
    plan_pose_index and the role (`stratum`) attached. `v8_only` returns the v8 index with
    its v8 roles (what `capture/plan_rebalance.py` plans from)."""
    absent = set(_load_json(ABSENT_LIST)["plan_pose_indices"])
    rows = [r for r in _v5_rows(only_ok)
            if not (r["capture_source"] == "part1" and int(r["plan_pose_index"]) in absent)]
    rows += _extension_rows(only_ok, v8_only) + _repair_rows(only_ok)
    inside = positions_inside_objects(rows)
    rows = [r for r in rows if r["position_key"] not in inside]
    rejected = _presence_rejects(rows)
    if rejected:
        raise RuntimeError(f"{len(rejected)} poses lie in robot-absent runs; re-capture them "
                           "before using the dataset")
    _check_unique(rows)
    # apply_partition=False gives the pool before any role, for capture/partition_v11.py
    return rows if v8_only or not apply_partition else _apply_partition(rows)


def _apply_partition(rows: list[dict]) -> list[dict]:
    """Set every row's role from the v11 partition; keep the previous role for provenance."""
    with PARTITION.open(newline="", encoding="utf-8") as handle:
        role = {r["position_key"]: r["role"] for r in csv.DictReader(handle)}
    present = {r["position_key"] for r in rows}
    if present - set(role) or set(role) - present:
        raise RuntimeError(f"v11 partition and captured positions differ: "
                           f"{len(present - set(role))} unassigned, {len(set(role) - present)} missing")
    for r in rows:
        r["v8_role"] = r["stratum"]
        r["stratum"] = role[r["position_key"]]
    return rows


def image_path(row: dict) -> Path:
    """Absolute path to a row's RGB image, in whichever capture pass it came from."""
    return dict((n, d) for n, d, _ in SOURCES)[row["capture_source"]] / row["image"]


def coverage() -> dict:
    rows = load_rows()
    positions = {r["position_key"]: r["stratum"] for r in rows}
    return {
        "rows": len(rows),
        "rows_by_source": dict(Counter(r["capture_source"] for r in rows)),
        "positions_by_role": dict(Counter(positions.values())),
        "unique_images_working": len({r["image_sha1"] for r in rows
                                      if r["stratum"] in ("D_mu", "D_R", "D_dev")}),
    }


if __name__ == "__main__":
    print(json.dumps(coverage(), indent=2))
