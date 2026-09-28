import json
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from ament_index_python.packages import get_package_share_directory
from unav_common.occlusion_geometry import (
    parse_collision_scene_from_world,
    parse_occlusion_scene_from_world,
    scene_to_json,
)


VALID_PLANNERS = {"visibility_aware_efe", "constant_R_efe"}


def load_world_profiles(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise RuntimeError(f"world_profiles not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    intrinsics = data.get("camera_intrinsics")
    if not isinstance(intrinsics, dict):
        raise RuntimeError("world_profiles.yaml must contain 'camera_intrinsics' mapping")

    worlds = data.get("worlds")
    if not isinstance(worlds, dict) or not worlds:
        raise RuntimeError("world_profiles.yaml must contain a non-empty 'worlds' mapping")

    _validate_intrinsics(intrinsics)
    for world_file, profile in worlds.items():
        _ensure_mapping(profile, f"worlds.{world_file}")
        local_intrinsics = profile.get("camera_intrinsics")
        if local_intrinsics is not None:
            _validate_intrinsics(local_intrinsics)
        visibility_defaults = profile.get("visibility_defaults")
        if visibility_defaults is not None:
            _validate_visibility_defaults(visibility_defaults)
        visibility_artifact = profile.get("visibility_artifact")
        if visibility_artifact is not None and not isinstance(visibility_artifact, str):
            raise RuntimeError(
                f"Expected 'worlds.{world_file}.visibility_artifact' to be a string path"
            )
    return {"camera_intrinsics": intrinsics, "worlds": worlds}



def get_worlds_dir() -> str:
    try:
        sim_share = get_package_share_directory("sim")
        world_dir = os.path.join(sim_share, "gazebo_worlds", "worlds")
        if os.path.isdir(world_dir):
            return world_dir
    except Exception:
        # fall through to local fallbacks
        pass

    # Fallback: prefer the repository-local source world directory when present.
    # File is located at: <repo>/src/experiments/experiments/core/world_profiles.py
    # parents[4] resolves to the repo root (UnembodiedNavigation).
    repo_root = Path(__file__).resolve().parents[4]
    candidate = repo_root / "src" / "sim" / "gazebo_worlds" / "worlds"
    if candidate.is_dir():
        return str(candidate)

    # Last-resort fallback: try a nearby relative path (defensive).
    candidate2 = Path(__file__).resolve().parents[3] / "sim" / "gazebo_worlds" / "worlds"
    if candidate2.is_dir():
        return str(candidate2)

    raise RuntimeError(
        "Could not locate 'sim' gazebo_worlds/worlds; install the 'sim' package or provide local src/sim/gazebo_worlds/worlds"
    )


def resolve_world_path(world_file: str) -> str:
    return os.path.join(get_worlds_dir(), world_file)


def _parse_world_info(world_path: str, camera_model: str) -> Tuple[List[str], str, List[float]]:
    try:
        tree = ET.parse(world_path)
    except ET.ParseError as exc:
        raise RuntimeError(f"Failed to parse world file '{world_path}': {exc}")
    root = tree.getroot()

    world_name = None
    for node in root.iter():
        if node.tag.endswith("world") and "name" in node.attrib:
            world_name = node.attrib.get("name")
            break

    models: List[str] = []
    camera_pose = None

    for include in root.iter():
        if not include.tag.endswith("include"):
            continue
        uri_node = None
        pose_node = None
        for child in list(include):
            if child.tag.endswith("uri"):
                uri_node = child
            elif child.tag.endswith("pose"):
                pose_node = child
        if uri_node is None or not uri_node.text:
            continue
        uri = uri_node.text.strip()
        if not uri.startswith("model://"):
            continue
        model_path = uri[len("model://"):]
        model_name = model_path.split("/")[0]
        if model_name:
            models.append(model_name)
        if model_name == camera_model:
            if pose_node is None or not pose_node.text:
                raise RuntimeError(
                    f"Camera include for '{camera_model}' in '{world_path}' is missing <pose>"
                )
            camera_pose = _parse_pose(pose_node.text)

    return models, world_name, camera_pose


def _parse_pose(text: str) -> List[float]:
    parts = [p for p in text.replace(",", " ").split() if p]
    if len(parts) != 6:
        raise RuntimeError("Camera pose must have 6 values: x y z roll pitch yaw")
    try:
        return [float(p) for p in parts]
    except ValueError as exc:
        raise RuntimeError(f"Invalid camera pose values: {exc}")


def _gz_resource_paths() -> List[str]:
    sim_share = get_package_share_directory("sim")
    sim_share_parent = os.path.dirname(sim_share)
    return [
        sim_share_parent,
        os.path.join(sim_share, "models"),
        os.path.join(sim_share, "gazebo_worlds", "models"),
        os.path.join(sim_share, "gazebo_worlds"),
        os.path.join(sim_share, "robot_description"),
        sim_share,
    ]


def _model_exists(model_name: str) -> bool:
    for base in _gz_resource_paths():
        if os.path.isdir(os.path.join(base, model_name)):
            return True
    return False


def _ensure_mapping(value: Any, name: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected '{name}' to be a mapping")
    return value


def _ensure_number(value: Any, name: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    raise RuntimeError(f"Expected '{name}' to be a number")


def _validate_intrinsics(intrinsics: Dict[str, Any]) -> None:
    _ensure_mapping(intrinsics, "camera_intrinsics")
    for key in ("img_width", "img_height", "fov_h_rad"):
        if key not in intrinsics:
            raise RuntimeError(f"camera_intrinsics missing '{key}'")
        _ensure_number(intrinsics[key], f"camera_intrinsics.{key}")


def _validate_visibility_defaults(visibility_defaults: Dict[str, Any]) -> None:
    _ensure_mapping(visibility_defaults, "visibility_defaults")
    numeric_keys = (
        "visibility_map_min_x",
        "visibility_map_max_x",
        "visibility_map_min_y",
        "visibility_map_max_y",
        "visibility_map_nx",
        "visibility_map_ny",
        "visibility_gp_length_scale",
        "visibility_gp_noise_var",
        "visibility_prior_occ",
        "visibility_beta",
        "visibility_target_height_m",
        "nogo_weight",
        "nogo_safe_distance",
        "nogo_logbarrier_eps",
        "nogo_warning_band",
        "nogo_near_weight",
    )
    for key in numeric_keys:
        if key in visibility_defaults:
            _ensure_number(visibility_defaults[key], f"visibility_defaults.{key}")


def validate_profile(
    world_file: str,
    profile: Dict[str, Any],
    world_path: str,
    camera_model: str,
) -> List[float]:
    _ensure_mapping(profile, world_file)
    if "world_name" not in profile:
        raise RuntimeError(f"Profile for '{world_file}' missing 'world_name'")
    if "spawn" not in profile:
        raise RuntimeError(f"Profile for '{world_file}' missing 'spawn'")
    if "planner_default" not in profile:
        raise RuntimeError(f"Profile for '{world_file}' missing 'planner_default'")

    spawn = _ensure_mapping(profile["spawn"], "spawn")
    for key in ("x", "y", "z", "yaw"):
        _ensure_number(spawn.get(key), f"spawn.{key}")

    planner = profile.get("planner_default")
    if planner not in VALID_PLANNERS:
        raise RuntimeError(
            f"Invalid planner_default '{planner}' for '{world_file}'. "
            f"Valid: {', '.join(sorted(VALID_PLANNERS))}"
        )

    if not os.path.isfile(world_path):
        raise RuntimeError(f"World file not found: {world_path}")

    included_models, parsed_world_name, camera_pose = _parse_world_info(
        world_path, camera_model
    )
    missing = [model for model in included_models if not _model_exists(model)]
    if missing:
        raise RuntimeError(
            f"World '{world_file}' references missing models: {', '.join(sorted(missing))}"
        )

    if camera_model not in included_models:
        raise RuntimeError(
            f"World '{world_file}' does not include '{camera_model}' but camera is required"
        )
    if camera_pose is None:
        raise RuntimeError(
            f"World '{world_file}' must specify <pose> for '{camera_model}' include"
        )
    if parsed_world_name and parsed_world_name != profile["world_name"]:
        raise RuntimeError(
            f"World '{world_file}' name '{parsed_world_name}' does not match profile "
            f"world_name '{profile['world_name']}'"
        )

    return camera_pose


def load_profile(path: str, world_file: str, camera_model: str = "external_camera") -> Tuple[Dict[str, Any], Dict[str, Any], str, List[float]]:
    data = load_world_profiles(path)
    profiles = data["worlds"]
    global_intrinsics = data["camera_intrinsics"]

    if world_file not in profiles:
        known = ", ".join(sorted(profiles.keys()))
        raise RuntimeError(
            f"No profile for world '{world_file}'. Available: {known or 'none'}"
        )
    profile = profiles[world_file]
    intrinsics = dict(global_intrinsics)
    local_intrinsics = profile.get("camera_intrinsics")
    if local_intrinsics is not None:
        intrinsics.update(local_intrinsics)
        _validate_intrinsics(intrinsics)
    world_path = resolve_world_path(world_file)
    camera_pose = validate_profile(world_file, profile, world_path, camera_model)
    return profile, intrinsics, world_path, camera_pose


def resolve_profile_asset_path(world_profiles_path: str, asset_path: str) -> str:
    raw_asset = str(asset_path or "").strip()
    if not raw_asset:
        return ""
    asset = Path(raw_asset)
    if asset.is_absolute():
        return str(asset)
    world_profiles_file = Path(world_profiles_path).expanduser().resolve()
    share_root = world_profiles_file.parent.parent if world_profiles_file.parent.name == "config" else world_profiles_file.parent
    return str((share_root / asset).resolve())


def parse_camera_pose_from_world(world_path: str, camera_model: str = "external_camera") -> List[float]:
    _, _, camera_pose = _parse_world_info(world_path, camera_model)
    if camera_pose is None:
        raise RuntimeError(
            f"World '{world_path}' must specify <pose> for '{camera_model}' include"
        )
    return camera_pose


def serialize_occlusion_geometry_from_world(
    world_path: str,
    model_name: str | Tuple[str, ...] = "warehouse_rack_occluders",
) -> str:
    scene = parse_occlusion_scene_from_world(world_path, model_name=model_name, geometry_tags=('collision',))
    return scene_to_json(scene)


def serialize_collision_geometry_from_world(
    world_path: str,
    model_names: Tuple[str, ...] = ("warehouse_walls", "warehouse_rack_occluders"),
    include_names: Tuple[str, ...] = (),
    profile: Dict[str, Any] | None = None,
    safety_margin_m: float = 0.10,
) -> str:
    scene = parse_collision_scene_from_world(
        world_path, model_names=model_names, include_names=include_names
    )
    payload = json.loads(scene_to_json(scene))
    margin = float(safety_margin_m)
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError("safety_margin_m must be finite and non-negative")
    for prism in payload.get("prisms", []):
        prism["xmin"] = float(prism["xmin"]) - margin
        prism["xmax"] = float(prism["xmax"]) + margin
        prism["ymin"] = float(prism["ymin"]) - margin
        prism["ymax"] = float(prism["ymax"]) + margin
    payload["geometric_safety_margin_m"] = margin
    # Mark the payload so consumers know these prisms are NON-traversable. Without
    # it, route seeding would treat them as a keep-in driveable union and emit no
    # routes at all.
    payload["model_name"] = "keepout_region"
    # Route seeding validates segments against this same keep-out geometry, so the
    # corridor axes must ride along with it.
    if profile is not None:
        for key in ("route_horizontal_centres", "route_vertical_centres"):
            values = profile.get(key)
            if values:
                payload[key] = [float(v) for v in values]
    return json.dumps(payload)


def serialize_driveable_geometry_from_profile(
    profile: Dict[str, Any], safety_margin_m: float = 0.10
) -> str:
    """Serialize the physical site boundary used by the hard keep-in gate.

    Historical versions returned the union of ``traversable`` lane annotations.
    Those annotations are route-seeding/visualisation aids, not physical safety
    authority.  The canonical free space is this site boundary minus the
    independently serialized collision scene.
    """
    margin = float(safety_margin_m)
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError("safety_margin_m must be finite and non-negative")
    regions = profile.get("known_2d_regions", []) or []
    prisms = []
    for r in regions:
        if str(r.get("type", "")).strip().lower() != "site_boundary":
            continue
        try:
            prisms.append({
                "name": str(r.get("name", "site_boundary")),
                "xmin": float(r["xmin"]) + margin,
                "xmax": float(r["xmax"]) - margin,
                "ymin": float(r["ymin"]) + margin,
                "ymax": float(r["ymax"]) - margin,
                "zmin": 0.0, "zmax": 0.1,
            })
        except (KeyError, TypeError, ValueError):
            continue
    if len(prisms) != 1:
        raise RuntimeError(
            "world profile must declare exactly one site_boundary region"
        )
    if any(p["xmin"] >= p["xmax"] or p["ymin"] >= p["ymax"] for p in prisms):
        raise RuntimeError("site_boundary collapses under geometric safety margin")
    payload = {
        "prisms": prisms,
        "model_name": "site_boundary",
        "geometric_safety_margin_m": margin,
    }
    # Corridor centre-lines for lane-graph route seeding. They travel with the
    # geometry because they cannot be inferred from an obstacle (keep-out) map.
    for key in ("route_horizontal_centres", "route_vertical_centres"):
        values = profile.get(key)
        if values:
            payload[key] = [float(v) for v in values]
    return json.dumps(payload)


def compute_look_at_from_pose(cam_pos: List[float], roll: float, pitch: float, yaw: float) -> List[float]:
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)

    # Gazebo camera include pose uses a convention where positive pitch points
    # the optical axis downward in world Z.
    forward = [cp * cy, cp * sy, -sp]
    if abs(forward[2]) < 1e-6:
        raise RuntimeError("Camera forward vector is parallel to ground plane")
    if forward[2] >= 0.0:
        raise RuntimeError("Camera must point downwards (negative z forward)")

    t = -cam_pos[2] / forward[2]
    if t <= 0.0:
        raise RuntimeError("Camera forward ray does not intersect ground plane in front")

    return [cam_pos[0] + t * forward[0], cam_pos[1] + t * forward[1], 0.0]


def compute_camera_quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> List[float]:
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return [qx, qy, qz, qw]
