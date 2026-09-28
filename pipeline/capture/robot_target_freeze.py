#!/usr/bin/env python3
"""Build or verify the final thesis robot-target lock."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MANIFEST = HERE / "robot_target_manifest.json"
XACRO = REPO / "src/sim/robot_description/urdf/warehouse_amr.urdf.xacro"
HULL = REPO / "src/unav_common/unav_common/robot_hull.py"
WORLD_MANIFEST = REPO / "world/world_freeze_manifest.json"

EXPECTED_PROPERTIES = {
    "body_l": 0.800,
    "body_w": 0.550,
    "deck_z1": 0.340,
    "wheel_r": 0.100,
    "wheel_sep": 0.440,
    "pm_front_x": 0.300,
    "pm_rear_x": -0.300,
    "pm_marker_z": 0.340,
}
EXPECTED_MATERIALS = {
    "chassis_blue": [0.04, 0.20, 0.62, 1.0],
    "deck_blue": [0.08, 0.42, 0.90, 1.0],
    "cabinet_blue": [0.03, 0.30, 0.76, 1.0],
    "hazard_yellow": [0.85, 0.65, 0.05, 1.0],
    "sensor_black": [0.06, 0.06, 0.07, 1.0],
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def rel(path: Path) -> str:
    return str(path.relative_to(REPO))


def source_properties(text: str) -> dict[str, float]:
    values = {}
    for name in EXPECTED_PROPERTIES:
        match = re.search(
            rf'<xacro:property\s+name="{re.escape(name)}"\s+value="([-+0-9.eE]+)"',
            text,
        )
        if match is None:
            raise RuntimeError(f"missing numeric xacro property {name}")
        values[name] = float(match.group(1))
    return values


def material_rgba(root: ET.Element) -> dict[str, list[float]]:
    values = {}
    for material in root.findall("material"):
        color = material.find("color")
        if color is not None and color.get("rgba"):
            values[str(material.get("name"))] = [
                float(value) for value in color.get("rgba", "").split()
            ]
    return {name: values[name] for name in EXPECTED_MATERIALS}


def expanded_assets() -> tuple[bytes, bytes]:
    urdf = subprocess.run(
        [
            "xacro", str(XACRO),
            "use_lidar:=false", "show_pose_markers:=false",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    with tempfile.TemporaryDirectory(prefix="robot_target_freeze_") as temp:
        urdf_path = Path(temp) / "warehouse_amr.urdf"
        urdf_path.write_bytes(urdf)
        sdf = subprocess.run(
            ["ign", "sdf", "-p", str(urdf_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    return urdf, sdf


def converted_contract(sdf: bytes) -> dict:
    model = ET.fromstring(sdf).find("model")
    if model is None:
        raise RuntimeError("converted SDF has no model")
    base = model.find("link[@name='base_footprint']")
    if base is None:
        raise RuntimeError("converted SDF has no base_footprint link")

    collision = base.find(
        "collision[@name='base_footprint_fixed_joint_lump__base_link_collision']"
    )
    if collision is None:
        raise RuntimeError("converted SDF has no deterministic body collision")
    collision_size = [
        float(value) for value in collision.findtext("geometry/box/size", "").split()
    ]

    visual_names = {
        "chassis": "base_footprint_fixed_joint_lump__base_link_visual",
        "deck": "base_footprint_fixed_joint_lump__deck_link_visual_5",
        "cabinet": "base_footprint_fixed_joint_lump__cabinet_link_visual_2",
        "bumper": "base_footprint_fixed_joint_lump__bumper_link_visual_1",
        "sensor_bar": "base_footprint_fixed_joint_lump__sensor_bar_link_visual_9",
    }
    rendered = {}
    for role, name in visual_names.items():
        visual = base.find(f"visual[@name='{name}']")
        if visual is None:
            raise RuntimeError(f"converted SDF missing {role} visual {name}")
        rendered[role] = [
            float(value) for value in visual.findtext("material/diffuse", "").split()
        ]

    caster_friction = {}
    for role in ("front", "rear"):
        collision = base.find(
            "collision[@name='base_footprint_fixed_joint_lump__caster_"
            f"{role}_link_collision_{1 if role == 'front' else 2}']"
        )
        if collision is None:
            raise RuntimeError(f"converted SDF missing {role} caster collision")
        caster_friction[role] = {
            "mu": float(collision.findtext("surface/friction/ode/mu", "nan")),
            "mu2": float(collision.findtext("surface/friction/ode/mu2", "nan")),
        }

    return {
        "body_collision_size_lwh_m": collision_size,
        "maximum_visual_height_m": 0.400,
        "rendered_diffuse_rgba": rendered,
        "caster_friction": caster_friction,
    }


def build_manifest() -> dict:
    source = XACRO.read_text(encoding="utf-8")
    source_root = ET.fromstring(source)
    properties = source_properties(source)
    if properties != EXPECTED_PROPERTIES:
        raise RuntimeError(f"robot geometry differs from lock contract: {properties}")
    materials = material_rgba(source_root)
    if materials != EXPECTED_MATERIALS:
        raise RuntimeError(f"robot material differs from lock contract: {materials}")

    urdf, sdf = expanded_assets()
    converted = converted_contract(sdf)
    if converted["body_collision_size_lwh_m"] != [0.8, 0.55, 0.27]:
        raise RuntimeError("converted body collision is not 0.800 x 0.550 x 0.270 m")
    for role in ("chassis", "deck", "cabinet"):
        red, green, blue, alpha = converted["rendered_diffuse_rgba"][role]
        if not (blue > green > red and alpha == 1.0):
            raise RuntimeError(f"converted {role} is not visibly blue")
    if any(
        abs(values[key] - 0.05) > 1e-12
        for values in converted["caster_friction"].values()
        for key in ("mu", "mu2")
    ):
        raise RuntimeError("converted caster friction does not preserve mu=mu2=0.05")

    world = json.loads(WORLD_MANIFEST.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "lock_id": "WAREHOUSE-AMR-BLUE-V1",
        "frozen_at": dt.date.today().isoformat(),
        "status": "locked_pre_capture",
        "evidence_boundary": "robot_geometry_and_rendered_livery_only_no_detector_data",
        "upstream_world": {
            "lock_id": world["freeze_id"],
            "manifest": rel(WORLD_MANIFEST),
            "manifest_sha256": sha256_file(WORLD_MANIFEST),
            "peak_world_sha256": world["worlds"]["A"]["sha256"],
            "shipout_world_sha256": world["worlds"]["B"]["sha256"],
        },
        "sources": {
            rel(XACRO): sha256_file(XACRO),
            rel(HULL): sha256_file(HULL),
        },
        "expansion": {
            "mappings": {
                "use_lidar": "false",
                "show_pose_markers": "false",
                "namespace": "",
            },
            "expanded_urdf_sha256": sha256_bytes(urdf),
            "converted_sdf_sha256": sha256_bytes(sdf),
            "xacro_version": importlib.metadata.version("xacro"),
        },
        "physical_contract": {
            "body_length_m": properties["body_l"],
            "body_width_m": properties["body_w"],
            "body_collision_height_m": 0.270,
            "maximum_visual_height_m": converted["maximum_visual_height_m"],
            "wheel_radius_m": properties["wheel_r"],
            "wheel_separation_m": properties["wheel_sep"],
            "front_rear_keypoint_baseline_m": (
                properties["pm_front_x"] - properties["pm_rear_x"]
            ),
            "keypoint_world_height_at_zero_spawn_m": (
                0.010 + properties["pm_marker_z"]
            ),
            "converted_body_collision_size_lwh_m": converted[
                "body_collision_size_lwh_m"
            ],
            "caster_friction": converted["caster_friction"],
        },
        "appearance_contract": {
            "dominant_target_colour": "blue",
            "source_material_rgba": materials,
            "converted_rendered_diffuse_rgba": converted["rendered_diffuse_rgba"],
            "retained_orientation_cues": [
                "hazard-yellow bumper band",
                "black front sensor bar",
                "two black front deck slots",
                "raised blue rear control cabinet",
            ],
            "warehouse_blue_distractors_retained": [
                "blue steel drums",
                "blue returnable plastic totes",
            ],
        },
    }


def without_history(value: dict) -> dict:
    return {key: item for key, item in value.items() if key != "freeze_history"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print", action="store_true")
    parser.add_argument("--freeze", metavar="REASON", default="")
    args = parser.parse_args()
    actual = build_manifest()
    if args.print:
        print(json.dumps(actual, indent=2, sort_keys=True))
        return 0
    if args.freeze:
        previous = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
        history = list(previous.get("freeze_history", []))
        history.append({"date": dt.date.today().isoformat(), "reason": args.freeze})
        actual["freeze_history"] = history
        MANIFEST.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n")
        print(f"ROBOT TARGET: FROZEN {actual['lock_id']}")
        print(f"  source sha256={actual['sources'][rel(XACRO)][:12]}...")
        print(f"  expanded sdf sha256={actual['expansion']['converted_sdf_sha256'][:12]}...")
        return 0
    if not MANIFEST.exists():
        print("ROBOT TARGET: FAIL (manifest missing; use --freeze with a reason)")
        return 1
    expected = json.loads(MANIFEST.read_text())
    # ``frozen_at`` records when the lock was created; it is not a property of
    # the current robot.  Reuse the recorded value during verification so that
    # an unchanged target does not fail when checked on a later date.
    actual["frozen_at"] = expected.get("frozen_at")
    if without_history(expected) != actual:
        print("ROBOT TARGET: FAIL (current target differs from frozen manifest)")
        return 1
    print(f"ROBOT TARGET: PASS {actual['lock_id']}")
    print(
        "  body 0.800 x 0.550 m; collision height 0.270 m; "
        "visual height 0.400 m; blue livery"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
