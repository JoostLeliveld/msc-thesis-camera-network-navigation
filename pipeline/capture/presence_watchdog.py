#!/usr/bin/env python3
"""Kill a running capture when the robot has vanished from the simulator.

A pose "trips" when at least MIN_IN_FRAME cameras should see the robot (its commanded pose
projects into their frame) but no camera's semantic mask contains it. Racks can hide every
heading of one position, so a trip run is only counted across positions: CONSECUTIVE
distinct positions in a row never trip in the good captures, while the broken v5 session
71daa8ab tripped thousands in a row. On such a run the
capture process is killed, so the retry loop restarts the simulator. Tripped poses are
re-captured afterwards from `validate_capture_presence`.

    python3 pipeline/capture/presence_watchdog.py CAPTURE_DIR
"""
from __future__ import annotations

import collections
import csv
import subprocess
import sys
import time
from pathlib import Path

MIN_IN_FRAME = 2
CONSECUTIVE = 3


def trips(rows: list[dict]) -> bool:
    in_frame = sum(r.get("nominal_in_frame") in ("1", "True") for r in rows)
    visible = sum(int(float(r.get("semantic_robot_pixels") or 0)) > 0 for r in rows)
    return in_frame >= MIN_IN_FRAME and visible == 0


def pose_trips(index: Path) -> list[tuple[str, int, str, bool]]:
    groups: dict[tuple[str, int], list[dict]] = collections.OrderedDict()
    with index.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("capture_status") != "ok":
                continue
            groups.setdefault((row["capture_session_id"], int(row["pose_id"])), []).append(row)
    return [(s, p, rows[0]["position_id"], trips(rows))
            for (s, p), rows in groups.items() if len(rows) == 5]


def main() -> int:
    index = Path(sys.argv[1]) / "capture_index.csv"
    killed_at = set()
    while True:
        if index.is_file():
            run, last_session = set(), None
            for session, pose, position, tripped in pose_trips(index):
                if session != last_session:
                    run, last_session = set(), session
                run = run | {position} if tripped else set()
                if len(run) >= CONSECUTIVE and (session, pose) not in killed_at:
                    killed_at.add((session, pose))
                    print(f"[{time.strftime('%H:%M:%S')}] robot absent at {len(run)} positions in a row "
                          f"(session {session[:8]}, pose {pose}); killing the capture", flush=True)
                    # SIGKILL: after a laptop sleep the capture ignored SIGTERM and kept writing.
                    subprocess.run(["pkill", "-9", "-f", "[c]apture_positions.py"], check=False)
        time.sleep(10)


if __name__ == "__main__":
    raise SystemExit(main())
