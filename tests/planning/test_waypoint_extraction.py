"""The global-to-local route handoff must honor its declared spacing."""

import numpy as np

from planning.planners.base_planner import extract_waypoints


def test_extract_waypoints_interpolates_long_global_steps():
    states = np.asarray([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 0.6, 1.0],
    ])
    points = np.asarray([[0.0, 0.0], *extract_waypoints(states, spacing_m=0.2)])
    gaps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    assert gaps.max() <= 0.2 + 1.0e-12
    np.testing.assert_allclose(points[-1], [1.0, 0.6])
