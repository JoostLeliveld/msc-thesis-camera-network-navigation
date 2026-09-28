from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from unav_common.preselected_route import (
    PreselectedRouteError,
    canonicalize_polyline_json,
    route_sha256,
    sample_polyline,
    validate_preselected_route,
)


def _driveable(*boxes: tuple[float, float, float, float]) -> str:
    return json.dumps({
        "model_name": "synthetic_driveable",
        "prisms": [
            {
                "name": f"lane_{index}",
                "xmin": xmin,
                "xmax": xmax,
                "ymin": ymin,
                "ymax": ymax,
                "zmin": 0.0,
                "zmax": 0.1,
            }
            for index, (xmin, xmax, ymin, ymax) in enumerate(boxes)
        ],
    })


def _source(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "route_geometry.json"
    path.write_bytes(b'{"frozen":"source artifact bytes"}')
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _validated(tmp_path: Path, route_text: str = "[[0.5,2],[2,2],[3.5,2]]"):
    _, canonical = canonicalize_polyline_json(route_text)
    source, source_sha = _source(tmp_path)
    return validate_preselected_route(
        route_text,
        route_sha256(canonical),
        start_xy=(0.5, 2.0),
        goal_xy=(3.5, 2.0),
        driveable_geometry_json=_driveable((0.0, 4.0, 0.0, 4.0)),
        declared_clearance_m=0.4,
        source_path=source,
        expected_source_sha256=source_sha,
        endpoint_tolerance_m=0.25,
        sample_step_m=0.04,
    )


def test_canonical_route_hash_and_provenance_are_deterministic(tmp_path: Path) -> None:
    route = _validated(tmp_path, " [ [ 0.5, 2 ], [2,2.0], [3.5,2] ] ")

    assert route.canonical_json == "[[0.5,2.0],[2.0,2.0],[3.5,2.0]]"
    assert route.sha256 == hashlib.sha256(route.canonical_json.encode()).hexdigest()
    assert route.points == ((0.5, 2.0), (2.0, 2.0), (3.5, 2.0))
    assert route.start_error_m == pytest.approx(0.0)
    assert route.goal_error_m == pytest.approx(0.0)
    assert route.minimum_driveable_clearance_m == pytest.approx(0.5)
    assert route.length_m == pytest.approx(3.0)
    assert route.provenance_dict()["validation_status"] == "passed"


@pytest.mark.parametrize(
    "payload, match",
    [
        ('{"routes": [[[0,0],[1,1]]]}', "exactly one JSON polyline"),
        ("[[0,0]]", "at least two"),
        ("[[0,0,1],[1,1]]", "exactly \[x,y\]"),
        ("[[0,0],[0,0]]", "duplicates"),
        ("[[true,0],[1,1]]", "finite JSON number"),
    ],
)
def test_polyline_parser_rejects_ambiguous_or_malformed_inputs(
    payload: str, match: str
) -> None:
    with pytest.raises(PreselectedRouteError, match=match):
        canonicalize_polyline_json(payload)


def test_route_and_source_hashes_are_both_fail_closed(tmp_path: Path) -> None:
    route_text = "[[0.5,2],[3.5,2]]"
    _, canonical = canonicalize_polyline_json(route_text)
    source, source_sha = _source(tmp_path)
    kwargs = dict(
        start_xy=(0.5, 2.0),
        goal_xy=(3.5, 2.0),
        driveable_geometry_json=_driveable((0.0, 4.0, 0.0, 4.0)),
        declared_clearance_m=0.4,
        source_path=source,
        expected_source_sha256=source_sha,
    )

    with pytest.raises(PreselectedRouteError, match="route SHA-256 mismatch"):
        validate_preselected_route(route_text, "0" * 64, **kwargs)

    kwargs["expected_source_sha256"] = "0" * 64
    with pytest.raises(PreselectedRouteError, match="source SHA-256 mismatch"):
        validate_preselected_route(route_text, route_sha256(canonical), **kwargs)


def test_registered_endpoint_tolerance_is_enforced(tmp_path: Path) -> None:
    with pytest.raises(PreselectedRouteError, match="registered task start"):
        _validated(tmp_path, "[[0.76,2],[2,2],[3.5,2]]")

    with pytest.raises(PreselectedRouteError, match="registered task goal"):
        _validated(tmp_path, "[[0.5,2],[2,2],[3.76,2]]")


def test_clearance_gate_walks_segments_not_only_vertices(tmp_path: Path) -> None:
    route_text = "[[0.5,2],[3.5,2]]"
    _, canonical = canonicalize_polyline_json(route_text)
    source, source_sha = _source(tmp_path)

    # Both vertices are deep inside a lane, but the straight segment crosses a
    # one-metre non-driveable gap. A point-only check would incorrectly pass.
    with pytest.raises(PreselectedRouteError, match="driveable clearance"):
        validate_preselected_route(
            route_text,
            route_sha256(canonical),
            start_xy=(0.5, 2.0),
            goal_xy=(3.5, 2.0),
            driveable_geometry_json=_driveable(
                (0.0, 1.5, 0.0, 4.0),
                (2.5, 4.0, 0.0, 4.0),
            ),
            declared_clearance_m=0.25,
            source_path=source,
            expected_source_sha256=source_sha,
        )


def test_segment_sampling_never_exceeds_registered_step() -> None:
    samples = sample_polyline([[0.0, 0.0], [0.11, 0.0], [0.11, 0.09]], maximum_step_m=0.04)
    distances = ((samples[1:] - samples[:-1]) ** 2).sum(axis=1) ** 0.5
    assert max(distances) <= 0.04 + 1.0e-12
    assert samples.tolist()[-1] == [0.11, 0.09]
