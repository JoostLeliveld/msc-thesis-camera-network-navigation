#!/usr/bin/env python3
"""Resume and complete the frozen Stage-05 resolution campaign unattended."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def completed(manifest: Path) -> bool:
    if not manifest.is_file():
        return False
    try:
        return json.loads(manifest.read_text(encoding="utf-8")).get("status") == "complete_pending_cross_resolution_evaluation"
    except (OSError, json.JSONDecodeError):
        return False


def run_logged(command: list[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--wait-for-imgsz", type=int, default=640)
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--maximum-wait-hours", type=float, default=6.0)
    args = parser.parse_args()
    protocol = args.protocol.resolve()
    stage = args.stage_root.resolve()
    stage.mkdir(parents=True, exist_ok=True)
    state_path = stage / "campaign_state.json"
    state = {"schema": "thesis_stage05_campaign.v1", "status": "running", "started_utc": utc(), "events": []}
    write_json(state_path, state)
    try:
        waiting = stage / f"imgsz{args.wait_for_imgsz}" / "trial_manifest.json"
        state["events"].append({"utc": utc(), "event": "waiting_for_existing_trial", "manifest": str(waiting)})
        write_json(state_path, state)
        wait_started = time.monotonic()
        while not completed(waiting):
            if time.monotonic() - wait_started > args.maximum_wait_hours * 3600:
                raise TimeoutError(f"Timed out waiting for existing trial: {waiting}")
            time.sleep(args.poll_seconds)
        state["events"].append({"utc": utc(), "event": "existing_trial_complete", "imgsz": args.wait_for_imgsz})
        write_json(state_path, state)

        reports = []
        for imgsz in (640, 960, 1280):
            trial_dir = stage / f"imgsz{imgsz}"
            manifest = trial_dir / "trial_manifest.json"
            if not completed(manifest):
                if trial_dir.exists():
                    raise RuntimeError(f"Refusing to overwrite incomplete trial directory: {trial_dir}")
                state["events"].append({"utc": utc(), "event": "training_started", "imgsz": imgsz})
                write_json(state_path, state)
                run_logged([
                    sys.executable, str(REPO / "pipeline/detector/train_trial.py"),
                    "--protocol", str(protocol), "--imgsz", str(imgsz), "--out", str(trial_dir),
                ], stage / f"imgsz{imgsz}_training.log")
                state["events"].append({"utc": utc(), "event": "training_complete", "imgsz": imgsz})
                write_json(state_path, state)

            report = trial_dir / "validation_report.json"
            if not report.is_file():
                state["events"].append({"utc": utc(), "event": "evaluation_started", "imgsz": imgsz})
                write_json(state_path, state)
                run_logged([
                    sys.executable, str(REPO / "pipeline/detector/evaluate_trial.py"),
                    "--protocol", str(protocol), "--trial", str(manifest), "--output", str(report),
                ], stage / f"imgsz{imgsz}_evaluation.log")
                state["events"].append({"utc": utc(), "event": "evaluation_complete", "imgsz": imgsz})
                write_json(state_path, state)
            reports.append(report)

        decision = stage / "resolution_selection.json"
        run_logged([
            sys.executable, str(REPO / "pipeline/detector/select_resolution.py"),
            "--protocol", str(protocol), "--reports", *(str(path) for path in reports),
            "--output", str(decision),
        ], stage / "resolution_selection.log")
        state["status"] = "complete_pending_review_and_lock"
        state["completed_utc"] = utc()
        state["decision"] = str(decision)
        state["events"].append({"utc": utc(), "event": "resolution_selected"})
        write_json(state_path, state)
        return 0
    except Exception as exc:
        state["status"] = "failed"
        state["failed_utc"] = utc()
        state["error"] = str(exc)
        state["traceback"] = traceback.format_exc()
        write_json(state_path, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
