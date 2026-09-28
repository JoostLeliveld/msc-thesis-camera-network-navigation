#!/usr/bin/env python3
"""Task-visibility rule (METHOD amendment 2026-09-25): in the dropout condition, the task's start
and goal must each be seen by at least one active camera in at least 90 % of the captured static
views within 0.75 m. A view counts as seen when the robot has semantic pixels in it.

    python3 pipeline/check_task_visibility.py                 check every task of a campaign config
    python3 pipeline/check_task_visibility.py --config C.yaml
    python3 pipeline/check_task_visibility.py --scan TASK x|y  scan the goal's lane in 0.1 m steps

Exits 1 when any start or goal fails, so config generation can refuse it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src/unav_common")]
from pipeline import dataset  # noqa: E402

RADIUS_M = 0.75
THRESHOLD = 0.90
DEFAULT_CONFIG = REPO / "logs/thesis/campaign_configs/campaign_seed91500.yaml"


class Views:
    def __init__(self):
        rows = dataset.load_rows()
        keys = {}
        view = np.array([keys.setdefault((r["position_key"], r["heading_id"], r["repetition_id"]), len(keys))
                         for r in rows])
        self.xy = np.zeros((len(keys), 2))
        self.xy[view] = [(float(r["robot_x"]), float(r["robot_y"])) for r in rows]
        self.cameras = sorted({r["camera_id"] for r in rows})
        self.seen = np.zeros((len(keys), len(self.cameras)), bool)
        for v, r in zip(view, rows):
            if float(r["semantic_robot_pixels"] or 0) > 0:
                self.seen[v, self.cameras.index(r["camera_id"])] = True

    def fractions(self, x: float, y: float) -> tuple[int, dict[str, float]]:
        near = np.hypot(self.xy[:, 0] - x, self.xy[:, 1] - y) <= RADIUS_M
        n = int(near.sum())
        return n, {c: float(self.seen[near, i].mean()) if n else 0.0 for i, c in enumerate(self.cameras)}

    def best_active(self, x: float, y: float, removed: str) -> tuple[int, float, str]:
        n, f = self.fractions(x, y)
        active = {c: v for c, v in f.items() if c != removed}
        cam = max(active, key=lambda c: active[c])
        return n, active[cam], cam


def tasks(config: Path | dict) -> list[dict]:
    cfg = yaml.safe_load(config.read_text()) if isinstance(config, Path) else config
    world = cfg.get("world", "warehouse_v2.world.sdf")
    spec = {t["name"]: t for t in yaml.safe_load((REPO / "pipeline/tasks.yaml").read_text())["tasks"][world]}
    return [{"name": n, "removed": t["condition_overrides"]["spatial_removal"]["removed_camera_id"],
             "start": spec[n]["start"], "goal": spec[n]["goal"]} for n, t in cfg["tasks"].items()]


def failures(config: Path | dict, views: Views | None = None) -> list[str]:
    """One line per start or goal that fails the rule; empty when every task passes."""
    views = views or Views()
    out = []
    for t in tasks(config):
        for label in ("start", "goal"):
            p = t[label]
            n, best, cam = views.best_active(p["x"], p["y"], t["removed"])
            if not (n > 0 and best >= THRESHOLD):
                out.append(f"{t['name']} {label} ({p['x']}, {p['y']}) without {t['removed']}: "
                           f"best {cam} {best:.2f} over {n} views")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--scan", nargs=2, metavar=("TASK", "AXIS"))
    ap.add_argument("--from", dest="lo", type=float, default=-9.25)
    ap.add_argument("--to", dest="hi", type=float, default=9.25)
    args = ap.parse_args()
    views = Views()
    task_list = tasks(args.config)
    if args.scan:
        task = next(t for t in task_list if t["name"] == args.scan[0])
        g = task["goal"]
        for v in np.arange(args.lo, args.hi + 1e-9, 0.1):
            x, y = (v, g["y"]) if args.scan[1] == "x" else (g["x"], v)
            n, best, cam = views.best_active(x, y, task["removed"])
            print(f"({x:7.3f},{y:7.3f}) views={n:4d} best={best:.2f} {cam}{'  PASS' if n and best >= THRESHOLD else ''}")
        return 0
    failed = False
    for t in task_list:
        for label in ("start", "goal"):
            p = t[label]
            n, best, cam = views.best_active(p["x"], p["y"], t["removed"])
            ok = n > 0 and best >= THRESHOLD
            failed |= not ok
            print(f"{'PASS' if ok else 'FAIL'} {t['name']} {label} ({p['x']}, {p['y']}) without {t['removed']}: "
                  f"best {cam} {best:.2f} over {n} views")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
