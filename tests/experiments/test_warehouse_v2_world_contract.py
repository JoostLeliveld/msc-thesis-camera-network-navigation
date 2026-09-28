from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import yaml


ROOT = Path(__file__).resolve().parents[2]
WORLD_DIR = ROOT / "src/sim/gazebo_worlds/worlds"
WORLD = WORLD_DIR / "warehouse_v2.world.sdf"
PROFILES = ROOT / "src/experiments/config/world_profiles.yaml"


def _profile() -> dict:
    return yaml.safe_load(PROFILES.read_text())["worlds"]["warehouse_v2.world.sdf"]


def test_profile_matches_the_world_name_and_tasks():
    tasks = yaml.safe_load((ROOT / "src/experiments/config/tasks.yaml").read_text())["tasks"]
    assert ET.parse(WORLD).getroot().find("world").get("name") == "warehouse_v2"
    assert _profile()["world_name"] == "warehouse_v2"
    assert _profile()["recommended_task"] in {t["name"] for t in tasks["warehouse_v2.world.sdf"]}


def test_profile_declares_the_five_camera_registry():
    expected_models = ["external_camera", "external_camera_b", "external_camera_c",
                       "external_camera_d", "external_camera_e"]
    profile = _profile()
    assert profile["camera_ids"] == [f"camera_{letter}" for letter in "ABCDE"]
    assert profile["camera_model_includes"] == expected_models
    world = ET.parse(WORLD).getroot().find("world")
    included = {inc.findtext("name"): inc.findtext("uri") for inc in world.findall("include")}
    assert {name: included[name] for name in expected_models} == {
        name: f"model://{name}" for name in expected_models}


def test_world_has_no_contact_sensors():
    """Collisions are scored offline as the footprint leaving the driveable region."""
    text = WORLD.read_text()
    assert 'type="contact"' not in text
    assert "gz-sim-contact-system" not in text


def test_every_collision_box_lies_inside_a_declared_zone_or_is_a_declared_object():
    """The SDF is the world; world/warehouse_v2.py describes it for planning and figures.

    Every rack and stack box of the occluder group must lie inside one of the declared
    zones, and the only boxes outside the zones are the profile's included loose objects,
    so the description cannot drift from the world unseen.
    """
    sys.path[:0] = [str(ROOT / "world"), str(ROOT / "src/unav_common")]
    import warehouse_v2
    from unav_common.occlusion_geometry import parse_collision_scene_from_world

    profile = _profile()
    zones = warehouse_v2.build().zones
    tol = 1e-3  # the SDF writes coordinates to the millimetre (0.5 mm rounding seen)

    def inside_a_zone(p) -> bool:
        return any(z.xmin - tol <= p.xmin and p.xmax <= z.xmax + tol
                   and z.ymin - tol <= p.ymin and p.ymax <= z.ymax + tol for z in zones)

    occluders = parse_collision_scene_from_world(
        str(WORLD), model_names=("warehouse_v2_occluders",), robot_z_range=(0.0, 0.55)).prisms
    assert occluders, "occluder group matched no geometry"
    outside = [p.name for p in occluders if not inside_a_zone(p)]
    assert not outside, f"collision boxes outside every declared zone: {outside}"

    included = parse_collision_scene_from_world(
        str(WORLD), model_names=(), include_names=tuple(profile["collision_include_names"]),
        model_search_paths=(str(ROOT / "src/sim/models"),), robot_z_range=(0.0, 0.55)).prisms
    assert {p.name.split("/")[0] for p in included} == set(profile["collision_include_names"])


def test_batch_mode_camera_set_comes_from_the_runtime_contract_not_a_literal():
    """The admissible camera set is the perception layer's own contract, and a mismatch is
    refused rather than trimmed."""
    source = (ROOT / "src/experiments/experiments/core/visibility_launch_common.py").read_text()
    assert "BATCHED_CAMERA_ORDER" in source
    assert "Refusing to silently omit cameras" in source
    assert "frozen_batch_ids" not in source

    spec = importlib.util.spec_from_file_location(
        "four_camera_runtime_contract",
        ROOT / "src/perception/perception/core/four_camera_runtime_contract.py")
    contract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contract)
    assert tuple(_profile()["camera_ids"]) == tuple(contract.BATCHED_CAMERA_ORDER)
