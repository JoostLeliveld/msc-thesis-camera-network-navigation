"""Canonical keep-in authority regressions (lanes are never safety geometry)."""
import json

import pytest

from experiments.core.world_profiles import serialize_driveable_geometry_from_profile


def test_site_boundary_is_used_instead_of_traversable_lane_union():
    profile = {'known_2d_regions': [
        {'name': 'site', 'type': 'site_boundary',
         'xmin': -5, 'xmax': 5, 'ymin': -4, 'ymax': 4},
        {'name': 'old_lane', 'type': 'traversable',
         'xmin': -1, 'xmax': 1, 'ymin': -4, 'ymax': 4},
    ]}
    payload = json.loads(serialize_driveable_geometry_from_profile(profile))
    assert payload['model_name'] == 'site_boundary'
    assert payload['geometric_safety_margin_m'] == pytest.approx(0.10)
    assert [item['name'] for item in payload['prisms']] == ['site']
    assert payload['prisms'][0] == {
        'name': 'site', 'xmin': -4.9, 'xmax': 4.9,
        'ymin': -3.9, 'ymax': 3.9, 'zmin': 0.0, 'zmax': 0.1,
    }


@pytest.mark.parametrize('regions', [[], [
    {'name': 'lane', 'type': 'traversable',
     'xmin': -1, 'xmax': 1, 'ymin': -1, 'ymax': 1},
]])
def test_missing_site_boundary_fails_closed(regions):
    with pytest.raises(RuntimeError, match='exactly one site_boundary'):
        serialize_driveable_geometry_from_profile({'known_2d_regions': regions})
