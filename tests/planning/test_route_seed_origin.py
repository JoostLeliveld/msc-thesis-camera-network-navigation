"""Map translation must not remove a corridor from the route seed family."""
import json

import numpy as np
import pytest

from unav_common.lane_graph_routes import (
    generate_diverse_route_candidates,
    generate_route_seeds,
)


def box(name, xmin, xmax, ymin, ymax):
    return dict(name=name, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=0., zmax=1.)


@pytest.mark.parametrize('origin_y', [0., -1., 1.])
def test_straight_corridor_has_a_route_at_every_map_origin(origin_y):
    geometry = json.dumps({'prisms': [box('corridor', -2., 2., origin_y-.5, origin_y+.5)]})
    seeds = generate_route_seeds(geometry, (-1., origin_y), (1., origin_y))
    assert len(seeds) == 1
    assert np.allclose(np.asarray(seeds[0]['waypoints']) - [0., origin_y], [[1., 0.]])


def test_zero_centred_corridor_is_between_existing_negative_and_positive_seeds():
    prisms = [box(f'cross_{y}', -2., 2., y-.4, y+.4) for y in (-2., 0., 2.)]
    prisms += [box('left_aisle', -1.4, -.6, -2.4, 2.4),
               box('right_aisle', .6, 1.4, -2.4, 2.4)]
    seeds = generate_route_seeds(json.dumps({'prisms': prisms}), (-1., -2.), (1., 2.))
    assert [s['name'] for s in seeds] == ['below_main_aisle', 'centre_cross_aisle', 'above_connector']
    assert seeds[1]['waypoints'] == [[-1., 0.], [1., 0.], [1., 2.]]
    assert seeds[0]['waypoints'] == [[1., -2.], [1., 2.]]
    assert seeds[2]['waypoints'] == [[-1., 2.], [1., 2.]]


def test_diverse_candidates_are_bounded_unique_and_geometry_only():
    prisms = [box(f'cross_{y}', -3., 3., y-.4, y+.4) for y in (-2., 0., 2.)]
    prisms += [box('left', -1.4, -.6, -2.4, 2.4),
               box('centre', -.4, .4, -2.4, 2.4),
               box('right', .6, 1.4, -2.4, 2.4)]
    geometry = json.dumps({'prisms': prisms})
    first = generate_diverse_route_candidates(
        geometry, (-1., -2.), (1., 2.), max_routes=5,
    )
    second = generate_diverse_route_candidates(
        geometry, (-1., -2.), (1., 2.), max_routes=5,
    )
    assert first == second
    assert 2 <= len(first) <= 5
    routes = [tuple(map(tuple, seed['waypoints'])) for seed in first]
    assert len(routes) == len(set(routes))


def test_diverse_candidates_remove_collinear_duplicate_waypoints_and_backtracks():
    prisms = [box(f'cross_{y}', -2., 2., y-.5, y+.5) for y in (-2., 0., 2.)]
    prisms += [box('middle', -.5, .5, -2.5, 2.5)]
    candidates = generate_diverse_route_candidates(
        json.dumps({'prisms': prisms}), (0., -1.5), (0., 1.5), max_routes=8,
    )
    assert candidates
    for candidate in candidates:
        points = [np.array([0., -1.5]), *map(np.asarray, candidate['waypoints'])]
        for a, b, c in zip(points, points[1:], points[2:]):
            first, second = b - a, c - b
            cross = first[0] * second[1] - first[1] * second[0]
            assert not (abs(cross) < 1e-9 and first @ second < 0.)
