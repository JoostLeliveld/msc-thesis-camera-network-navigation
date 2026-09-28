from pathlib import Path

from unav_common.occlusion_geometry import parse_collision_scene_from_world


ROOT = Path(__file__).resolve().parents[2]
WORLD = ROOT / 'src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf'
PROPS = ('forklift_parked', 'pallet_jack', 'bin_office',
         'pallet_loose_1', 'pallet_loose_2')


def test_active_world_declared_props_enter_planner_collision_scene():
    scene = parse_collision_scene_from_world(
        str(WORLD),
        model_names=('warehouse_shell', 'warehouse_v2_occluders'),
        include_names=PROPS,
    )
    names = {prism.name.split('/', 1)[0] for prism in scene.prisms}
    assert set(PROPS) <= names
    forklift = [p for p in scene.prisms if p.name.startswith('forklift_parked/')]
    assert forklift
    assert not any('overhead_guard' in p.name for p in forklift)
    assert len([p for p in scene.prisms if p.name.startswith('pallet_loose_')]) == 4


def test_included_prop_meshes_are_resolved_at_robot_height():
    # pallet_jack carries a mesh collision, so this covers the mesh path.
    scene = parse_collision_scene_from_world(
        str(WORLD), model_names=(), include_names=('pallet_jack',)
    )
    assert {p.name.split('/', 1)[0] for p in scene.prisms} == {'pallet_jack'}
    for prism in scene.prisms:
        assert prism.xmax > prism.xmin
        assert prism.ymax > prism.ymin
        assert prism.zmax >= 0.07
        assert prism.zmin <= 0.35
