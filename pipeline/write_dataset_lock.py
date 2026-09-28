#!/usr/bin/env python3
"""(Re)write the dataset lock (v11) from the index: roles, counts and source hashes.

The index itself (dataset.load_rows) refuses any robot-absent run, so a
lock is only written for a dataset that passed the presence check.

    python3 pipeline/write_dataset_lock.py
"""
from __future__ import annotations

import collections
import copy
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from pipeline import dataset as ds  # noqa: E402

HERE = REPO / "pipeline"
LOCK = HERE / "dataset_lock.json"
ROOT = "logs/thesis"
CAPTURES = f"{ROOT}/captures"
CHECKPOINT = f"{ROOT}/detector/training/imgsz960/upper_finetune/weights/best.pt"


def sha(rel: str) -> str:
    return hashlib.sha256((REPO / rel).read_bytes()).hexdigest()


def rel(path: Path) -> str:
    # absolute, not resolved: logs/ may be a symlink into another checkout's data
    return str(Path(os.path.abspath(path)).relative_to(REPO))


def main() -> int:
    rows = ds.load_rows()
    work = [r for r in rows if r["stratum"] in ("D_mu", "D_R", "D_dev")]
    audit = [r for r in rows if r["stratum"] == "final_audit"]
    positions: dict[str, dict] = {}
    for r in rows:
        p = positions.setdefault(r["position_key"], {
            "position_key": r["position_key"], "x": round(float(r["robot_x"]), 4),
            "y": round(float(r["robot_y"]), 4), "role": r["stratum"],
            "source": r["capture_source"], "v8_role": r["v8_role"], "camera_opportunities": 0})
        p["camera_opportunities"] += 1
    table = f"{CAPTURES}/v11/capture_positions_v11.csv"
    with (REPO / table).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(next(iter(positions.values()))))
        writer.writeheader()
        writer.writerows(sorted(positions.values(), key=lambda p: p["position_key"]))
    roles = collections.Counter(p["role"] for p in positions.values())

    # Fixed sections (capture environment, detector, prohibitions) carry over from the
    # current lock; everything that depends on the data is recomputed below.
    lock = copy.deepcopy(json.loads(LOCK.read_text()))
    lock.pop("derived_from", None)
    lock.update({
        "lock_id": "THESIS-REFERENCE-POSITION-DATASET-V11", "status": "locked_before_refit",
        "campaign_root": ROOT, "amendment": "docs/METHOD.md, Amendment 2026-09-25 (afternoon): one dataset, re-split (v11)",
        "audit_command": "python3 pipeline/audit_dataset.py",
    })
    lock["detector"]["checkpoint"] = CHECKPOINT
    if sha(CHECKPOINT) != lock["detector"]["checkpoint_sha256"]:
        raise RuntimeError("frozen detector checkpoint hash differs from the lock")
    lock["sampling"] = {
        "loader": "pipeline/dataset.py", "loader_sha256": sha("pipeline/dataset.py"),
        "positions": table, "positions_sha256": sha(table),
        "topup_rule": "pipeline/capture/plan_topup.py",
        "partition_rule": "pipeline/capture/partition_v11.py",
        "partition_rule_sha256": sha("pipeline/capture/partition_v11.py"),
        "strata": f"{CAPTURES}/v9/strata_v9.json", "strata_sha256": sha(f"{CAPTURES}/v9/strata_v9.json"),
        "partition": rel(ds.PARTITION), "partition_sha256": sha(rel(ds.PARTITION)),
        "robot_absent_list": rel(ds.ABSENT_LIST), "robot_absent_list_sha256": sha(rel(ds.ABSENT_LIST)),
        "capture_passes": {
            name: {"capture_index": rel(directory / "capture_index.csv"),
                   "capture_index_sha256": sha(rel(directory / "capture_index.csv")),
                   "pose_file": rel(poses), "pose_file_sha256": sha(rel(poses))}
            for name, directory, poses in ds.SOURCES},
        "position_count": len(positions), "expected_camera_opportunities": len(rows),
        "expected_pose_batches": len(rows) // 5,
    }
    lock["partition"]["roles"] = dict(sorted(roles.items()))
    lock["partition"].pop("added_position_rule", None)
    lock["partition"]["assignment"] = (
        "capture/partition_v11.py: every captured position re-split as one dataset; inside "
        "each v9 2 x 2 m stratum positions are ordered by sha256(seed, key) regardless of "
        "capture; final_audit, D_dev and D_R take area-proportional quotas, the rest D_mu; "
        "no captured position is dropped")
    lock["opportunity_accounting"].update({
        "expected_working_opportunities": len(work),
        "expected_working_unique_images": len({r["image_sha1"] for r in work}),
        "expected_final_audit_opportunities": len(audit),
    })
    LOCK.write_text(json.dumps(lock, indent=1) + "\n")
    print(LOCK.name, dict(sorted(roles.items())), len(rows), len(work), len(audit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
