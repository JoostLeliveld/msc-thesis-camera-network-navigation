#!/usr/bin/env python3
"""Lock every final-campaign input by path and SHA-256 before the first run.

Writes logs/thesis/final_campaign/campaign/manifest.json: the code commit (a dirty tree is refused), the
dataset lock, the gate, the detector, every fit artifact, the world, the tasks, the three
per-seed configs and the 30 solved routes. Re-running it after the campaign has started
compares instead of overwriting, so any drift is reported.

    python3 pipeline/campaign_manifest.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
R = REPO / "logs/thesis"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs(campaign_root: Path) -> dict[str, str]:
    files = [REPO / p for p in (
        "pipeline/dataset_lock.json", "config/sensor_gate.yaml", "pipeline/tasks.yaml",
        "pipeline/execution_template.yaml", "pipeline/route_planning_template.yaml",
        "src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf",
        "src/experiments/config/world_profiles.yaml",
        "logs/thesis/detector/training/imgsz960/upper_finetune/weights/best.pt",
        "logs/thesis/final_audit_protocol.json")]
    for sub in ("runtime_r012", "planning_precision", "covariance", "correction"):
        files += sorted(p for p in (R / "fits" / sub).rglob("*") if p.is_file())
    files += sorted((campaign_root / "campaign_configs").glob("*.yaml"))
    files += sorted((campaign_root / "routes").glob("*/*.route.json"))
    files += sorted((campaign_root / "routes").glob("*/*.npz"))
    replay = campaign_root / "routes/follower_replay_check.json"
    if not replay.is_file():
        raise FileNotFoundError(f"missing final route replay: {replay}")
    replay_report = json.loads(replay.read_text())
    if replay_report.get("failing") != 0 or len(replay_report.get("routes", [])) != 30:
        raise RuntimeError("final route replay is not a 30-route pass")
    files.append(replay)
    return {str(p.relative_to(REPO)): sha(p) for p in files}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, default=R / "final_campaign",
                        help="Root containing campaign_configs/, routes/ and campaign/.")
    args = parser.parse_args()
    campaign_root = args.campaign_root.resolve()
    try:
        campaign_root.relative_to(REPO)
    except ValueError:
        print(f"refusing campaign root outside the repository: {campaign_root}", file=sys.stderr)
        return 1
    out = campaign_root / "campaign/manifest.json"
    git = lambda *a: subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty:
        print(f"refusing: tracked changes in the working tree\n{dirty}", file=sys.stderr)
        return 1
    manifest = {"schema": "thesis_campaign_manifest.v1", "commit": git("rev-parse", "HEAD"),
                "campaign_root": str(campaign_root.relative_to(REPO)),
                "inputs": inputs(campaign_root)}
    n_routes = sum(k.endswith(".route.json") for k in manifest["inputs"])
    if n_routes != 30:
        print(f"refusing: expected 30 solved routes, found {n_routes}", file=sys.stderr)
        return 1
    if out.exists():
        old = json.loads(out.read_text())
        drift = sorted(k for k in set(old["inputs"]) | set(manifest["inputs"])
                       if old["inputs"].get(k) != manifest["inputs"].get(k))
        drift += ["commit"] if old["commit"] != manifest["commit"] else []
        drift += (["campaign_root"] if old.get("campaign_root") != manifest["campaign_root"] else [])
        print(json.dumps({"manifest": str(out), "drift": drift}, indent=1))
        return 1 if drift else 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({"manifest": str(out), "files": len(manifest["inputs"]), "commit": manifest["commit"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
