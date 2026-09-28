import json

from unav_common.lane_graph_routes import _route_centres_from_geometry
from unav_common.occlusion_geometry import scene_from_json


def test_declared_route_centres_override_decomposition_cells():
    geometry = json.dumps({
        "route_horizontal_centres": [-2.0, 3.0],
        "route_vertical_centres": [-4.0, 5.0],
        "prisms": [
            {"name": "cell", "xmin": 0.0, "xmax": 10.0,
             "ymin": 0.0, "ymax": 1.0, "zmin": 0.0, "zmax": 0.1},
        ],
    })
    prisms = scene_from_json(geometry).prisms
    horizontal, vertical = _route_centres_from_geometry(geometry, prisms)
    assert horizontal == [-2.0, 3.0]
    assert vertical == [-4.0, 5.0]


def test_historical_geometry_still_infers_axes():
    geometry = json.dumps({"prisms": [
        {"name": "h", "xmin": 0.0, "xmax": 4.0,
         "ymin": 0.0, "ymax": 1.0, "zmin": 0.0, "zmax": 0.1},
        {"name": "v", "xmin": 5.0, "xmax": 6.0,
         "ymin": 0.0, "ymax": 4.0, "zmin": 0.0, "zmax": 0.1},
    ]})
    prisms = scene_from_json(geometry).prisms
    horizontal, vertical = _route_centres_from_geometry(geometry, prisms)
    assert horizontal == [0.5]
    assert vertical == [5.5]
