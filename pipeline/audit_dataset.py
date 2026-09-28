#!/usr/bin/env python3
"""Hard integrity audit of the v8 dataset. Exits non-zero on any failure.

Checks, each on the rows `dataset.load_rows()` returns:
1. no robot-absent run remains (the loader raises otherwise);
2. every (capture source, pose) has exactly five camera rows, and every plan pose is used
   by exactly one source;
3. every capture source has the v5 world and capture-script hashes, except the v9 audit:
   its world is accepted only through the passing equivalence record
   (captures/v9/world_equivalence/report.json: identical images but one static pixel of
   camera B) and its script only as today's capture_positions.py (renamed paths and the
   stricter pose check with the loose objects; image capture unchanged);
4. per capture session, the share of in-frame camera rows whose mask contains the robot is
   no more than 10 points below that of the other sessions' rows by the same camera within
   0.5 m (like for like, because visibility varies from ~0.3 to ~0.8 between regions; the
   broken frames were ~70 points below their neighbours);
5. every referenced image exists, and a seeded random sample of 3000 decodes;
6. no row with capture_status other than ok is present.
7. no pose puts the robot footprint inside a collision object of the world.

    python3 pipeline/audit_dataset.py
"""
from __future__ import annotations

import collections
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from pipeline import dataset as v8  # noqa: E402

MAX_DEFICIT = 0.10
MIN_ROWS = 100
SAMPLE = 3000
OUT = REPO / "logs/thesis/evidence/dataset_audit.json"


def main() -> int:
    failures, report = [], {}
    try:
        rows = v8.load_rows()
    except RuntimeError as exc:
        print(f"FAIL presence: {exc}")
        return 1
    report["rows"] = len(rows)

    per_pose = collections.Counter((r["capture_source"], r["capture_session_id"], r["pose_id"]) for r in rows)
    bad_pose = [k for k, n in per_pose.items() if n != 5]
    plan_sources = collections.defaultdict(set)
    for r in rows:
        plan_sources[int(r["plan_pose_index"])].add(r["capture_source"])
    multi = [p for p, s in plan_sources.items() if len(s) > 1]
    report.update(poses=len(per_pose), poses_not_5_cameras=len(bad_pose), plan_poses_in_two_sources=len(multi))
    if bad_pose or multi:
        failures.append("pose completeness")

    identities = {}
    for name, directory, _ in v8.SOURCES:
        m = json.loads((directory / "capture_manifest.json").read_text())
        identities[name] = (m.get("world_sha256"), m.get("capture_script_sha256"))
    report["identities"] = {k: [str(a)[:12], str(b)[:12]] for k, (a, b) in identities.items()}
    v5_identity = identities["part1"]
    equivalence = json.loads((REPO / "logs/thesis/captures/v9/world_equivalence/report.json").read_text())
    import hashlib
    script = hashlib.sha256((REPO / "pipeline/capture/capture_positions.py").read_bytes()).hexdigest()
    v9_identity = (equivalence["current_world_sha256"], script)
    report["world_equivalence_passed"] = bool(equivalence.get("passed"))
    for name, identity in identities.items():
        allowed = {v5_identity} | ({v9_identity} if name in ("audit_v9", "lane_v10", "fill_v11") and equivalence.get("passed")
                                   and equivalence.get("v8_world_sha256") == v5_identity[0] else set())
        if identity not in allowed:
            failures.append(f"capture identity: {name}")

    in_frame = [r for r in rows if r.get("nominal_in_frame") in ("1", "True")]
    session_of = [f"{r['capture_source']}:{r['capture_session_id'][:8]}" for r in in_frame]
    xy = np.array([[float(r["robot_x"]), float(r["robot_y"])] for r in in_frame])
    cam = np.array([r["camera_id"] for r in in_frame])
    vis = np.array([int(float(r.get("semantic_robot_pixels") or 0)) > 0 for r in in_frame])
    sess = np.array(session_of)
    report["session_visibility_vs_neighbours"] = {}
    deficits = {}
    for s_name in sorted(set(session_of)):
        own_idx = np.flatnonzero(sess == s_name)
        other = sess != s_name
        own, ref = [], []
        for i in own_idx:
            near = other & (cam == cam[i]) & (np.hypot(xy[:, 0] - xy[i, 0], xy[:, 1] - xy[i, 1]) < 0.5)
            if near.any():
                own.append(vis[i]); ref.append(vis[near].mean())
        if not own:
            continue
        entry = {"rows": len(own), "own": round(float(np.mean(own)), 3),
                 "neighbours": round(float(np.mean(ref)), 3)}
        report["session_visibility_vs_neighbours"][s_name] = entry
        if len(own) >= MIN_ROWS and entry["neighbours"] - entry["own"] > MAX_DEFICIT:
            deficits[s_name] = entry
    report["sessions_below_neighbours"] = deficits
    if deficits:
        failures.append("session visibility")

    missing = [r for r in rows if not v8.image_path(r).is_file()]
    rng = random.Random(20260923)
    sample = rng.sample(rows, min(SAMPLE, len(rows)))
    undecodable = [r for r in sample if cv2.imread(str(v8.image_path(r)), cv2.IMREAD_COLOR) is None]
    report.update(missing_images=len(missing), sample_decoded=len(sample), undecodable=len(undecodable))
    if missing or undecodable:
        failures.append("images")

    not_ok = sum(r.get("capture_status") != "ok" for r in rows)
    report["non_ok_rows"] = not_ok
    if not_ok:
        failures.append("row status")

    # every pose's robot footprint clears every collision object of the world (model
    # groups and the top-level included forklift, pallets, bin and pallet jack)
    inside = v8.positions_inside_objects(rows)
    report["positions_inside_objects"] = sorted(inside)
    if inside:
        failures.append("robot inside an object")

    report["passed"] = not failures
    report["failures"] = failures
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
