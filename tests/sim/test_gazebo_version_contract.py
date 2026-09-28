from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]


def test_humble_launch_pins_fortress_and_both_resource_path_spellings():
    launch = (ROOT / "src/sim/launch/gazebo.launch.py").read_text(encoding="utf-8")

    assert '"gz_version": "6"' in launch
    assert 'name="GZ_SIM_RESOURCE_PATH"' in launch
    assert 'name="IGN_GAZEBO_RESOURCE_PATH"' in launch


def test_sim_package_describes_the_runtime_it_actually_uses():
    root = ET.parse(ROOT / "src/sim/package.xml").getroot()
    description = " ".join((root.findtext("description") or "").split())

    assert "Fortress" in description
    assert "Sim 6" in description
    assert "Harmonic" not in description


def test_set_pose_bridge_uses_service_syntax_and_unsafe_reset_is_not_bridged():
    launch = (ROOT / "src/sim/launch/bringup_sim.launch.py").read_text(encoding="utf-8")

    assert "/set_pose@ros_gz_interfaces/srv/SetEntityPose'" in launch
    assert "/control@ros_gz_interfaces/srv/ControlWorld'" not in launch
    assert "SetEntityPose@gz.msgs" not in launch
    assert "ControlWorld@gz.msgs" not in launch


def test_optional_extra_camera_bridges_cover_e_through_l():
    launch = (ROOT / "src/sim/launch/bringup_sim.launch.py").read_text(encoding="utf-8")

    assert 'extra_camera_suffixes = tuple("efghijkl")' in launch
    assert 'f"bridge_camera_{suffix}"' in launch
    assert 'f"bridge_segmentation_{suffix}"' in launch
    assert 'f"/external_camera_{suffix}/image_raw"' in launch
    assert 'f"/external_camera_{suffix}/segmentation/labels_map"' in launch
