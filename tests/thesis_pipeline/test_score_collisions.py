"""Known-answer checks for the offline collision score (footprint vs driveable region)."""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from pipeline.score_collisions import DriveableRegion, score_poses  # noqa: E402


def _box(xmin, xmax, ymin, ymax, name="box"):
    return SimpleNamespace(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, name=name)


def _region(*obstacles):
    return DriveableRegion(list(obstacles) or [_box(100, 101, 100, 101)], _box(0, 10, 0, 10),
                           length=0.8, width=0.55)


def _line(x0, y0, x1, y1, yaw, n):
    return [(float(i), x0 + (x1 - x0) * i / (n - 1), y0 + (y1 - y0) * i / (n - 1), yaw)
            for i in range(n)]


def test_a_clear_path_does_not_collide():
    result = score_poses(_region(_box(4, 6, 4, 6)), _line(1, 1, 9, 1, 0.0, 50))
    assert result["collision"] is False
    assert result["min_obstacle_clearance_m"] > 0 and result["min_boundary_clearance_m"] > 0


def test_driving_into_an_obstacle_collides_at_the_first_overlapping_sample():
    result = score_poses(_region(_box(4, 6, 4, 6)), _line(5, 1, 5, 5, math.pi / 2, 41))
    assert result["collision"] is True and result["first_exit"]["kind"] == "obstacle"
    # front edge reaches y = 4 when the centre reaches 3.6
    assert 3.55 <= result["first_exit"]["y"] <= 3.65


def test_leaving_the_site_boundary_collides():
    result = score_poses(_region(), _line(5, 1, 9.9, 1, 0.0, 50))
    assert result["collision"] is True and result["first_exit"]["kind"] == "boundary"
    assert 9.55 <= result["first_exit"]["x"] <= 9.7


def test_a_crossing_between_two_clear_samples_is_caught():
    thin = _box(4.9, 5.1, 0, 3)
    poses = [(0.0, 3.0, 1.0, 0.0), (1.0, 7.0, 1.0, 0.0)]
    region = _region(thin)
    assert all(min(region.clearance(p[1:])) > 0 for p in poses)
    result = score_poses(region, poses)
    assert result["collision"] is True and result["first_exit"]["kind"] == "between_samples"


def test_real_world_known_poses():
    region = DriveableRegion.for_world()
    # task A start, in the open north-west corridor
    assert min(region.clearance((-10.7, 8.55, 0.0))) > 0
    # centre of rack block A1
    assert region.clearance((-8.9, 1.5, 0.0))[0] < 0
