"""Shared occlusion geometry helpers for warehouse-style visibility models."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Iterable, Sequence
import xml.etree.ElementTree as ET

import numpy as np


_EMPTY_HEIGHT = -1.0e6


@dataclass(frozen=True)
class AxisAlignedPrism:
    """Axis-aligned vertical prism used for fast occlusion queries."""

    name: str
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float

    @property
    def center_xy(self) -> np.ndarray:
        return np.array([
            0.5 * (self.xmin + self.xmax),
            0.5 * (self.ymin + self.ymax),
        ], dtype=float)

    @property
    def size_xy(self) -> np.ndarray:
        return np.array([
            max(self.xmax - self.xmin, 1e-6),
            max(self.ymax - self.ymin, 1e-6),
        ], dtype=float)

    def contains_xy(self, x: float, y: float) -> bool:
        return (self.xmin <= x <= self.xmax) and (self.ymin <= y <= self.ymax)

    def signed_distance_xy(self, xy: np.ndarray) -> np.ndarray:
        pts = np.asarray(xy, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(1, 2)
        dx = np.maximum(np.maximum(self.xmin - pts[:, 0], 0.0), pts[:, 0] - self.xmax)
        dy = np.maximum(np.maximum(self.ymin - pts[:, 1], 0.0), pts[:, 1] - self.ymax)
        outside = np.hypot(dx, dy)

        inside_x = np.minimum(pts[:, 0] - self.xmin, self.xmax - pts[:, 0])
        inside_y = np.minimum(pts[:, 1] - self.ymin, self.ymax - pts[:, 1])
        inside_depth = np.minimum(inside_x, inside_y)
        inside = (dx <= 0.0) & (dy <= 0.0)
        signed = outside
        signed[inside] = -inside_depth[inside]
        return signed

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'xmin': float(self.xmin),
            'xmax': float(self.xmax),
            'ymin': float(self.ymin),
            'ymax': float(self.ymax),
            'zmin': float(self.zmin),
            'zmax': float(self.zmax),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'AxisAlignedPrism':
        return cls(
            name=str(data.get('name', 'prism')),
            xmin=float(data['xmin']),
            xmax=float(data['xmax']),
            ymin=float(data['ymin']),
            ymax=float(data['ymax']),
            zmin=float(data['zmin']),
            zmax=float(data['zmax']),
        )


@dataclass(frozen=True)
class OcclusionScene:
    prisms: tuple[AxisAlignedPrism, ...]
    source_world: str = ''
    model_name: str = 'warehouse_rack_occluders'

    def to_dict(self) -> dict:
        return {
            'source_world': self.source_world,
            'model_name': self.model_name,
            'prisms': [prism.to_dict() for prism in self.prisms],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(',', ':'))

    @classmethod
    def from_dict(cls, data: dict) -> 'OcclusionScene':
        prisms = tuple(AxisAlignedPrism.from_dict(item) for item in data.get('prisms', []))
        return cls(
            prisms=prisms,
            source_world=str(data.get('source_world', '')),
            model_name=str(data.get('model_name', 'warehouse_rack_occluders')),
        )

    @classmethod
    def from_json(cls, text: str | None) -> 'OcclusionScene':
        if text is None:
            return cls(prisms=())
        payload = str(text).strip()
        if not payload:
            return cls(prisms=())
        return cls.from_dict(json.loads(payload))


@dataclass(frozen=True)
class _Pose6D:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0


def _tag_matches(node: ET.Element, name: str) -> bool:
    return node.tag == name or node.tag.endswith(name)


def _find_child(node: ET.Element, name: str) -> ET.Element | None:
    for child in list(node):
        if _tag_matches(child, name):
            return child
    return None


def _parse_pose_node(node: ET.Element | None) -> _Pose6D:
    if node is None or not (node.text and node.text.strip()):
        return _Pose6D()
    parts = [p for p in node.text.replace(',', ' ').split() if p]
    if len(parts) != 6:
        raise RuntimeError(f'Expected pose with 6 values, got {len(parts)}')
    vals = [float(part) for part in parts]
    return _Pose6D(*vals)


def _compose_pose(base: _Pose6D, local: _Pose6D) -> _Pose6D:
    cy = math.cos(base.yaw)
    sy = math.sin(base.yaw)
    x = base.x + cy * local.x - sy * local.y
    y = base.y + sy * local.x + cy * local.y
    z = base.z + local.z
    return _Pose6D(
        x=x,
        y=y,
        z=z,
        roll=base.roll + local.roll,
        pitch=base.pitch + local.pitch,
        yaw=base.yaw + local.yaw,
    )


def _parse_box_prism(name: str, pose: _Pose6D, size_text: str) -> AxisAlignedPrism:
    sx, sy, sz = [float(part) for part in size_text.replace(',', ' ').split() if part]
    cy = abs(math.cos(pose.yaw))
    syaw = abs(math.sin(pose.yaw))
    hx = 0.5 * (cy * sx + syaw * sy)
    hy = 0.5 * (syaw * sx + cy * sy)
    hz = 0.5 * sz
    return AxisAlignedPrism(
        name=name,
        xmin=pose.x - hx,
        xmax=pose.x + hx,
        ymin=pose.y - hy,
        ymax=pose.y + hy,
        zmin=pose.z - hz,
        zmax=pose.z + hz,
    )


def _parse_cylinder_prism(name: str, pose: _Pose6D, radius_text: str, length_text: str) -> AxisAlignedPrism:
    radius = float(radius_text)
    length = float(length_text)
    hz = 0.5 * length
    return AxisAlignedPrism(
        name=name,
        xmin=pose.x - radius,
        xmax=pose.x + radius,
        ymin=pose.y - radius,
        ymax=pose.y + radius,
        zmin=pose.z - hz,
        zmax=pose.z + hz,
    )


def _geometry_prisms_from_node(
    *,
    model_name: str,
    link_name: str,
    geometry_node: ET.Element | None,
    pose: _Pose6D,
    element_name: str,
    model_search_paths: Sequence[str] = (),
    robot_z_range: tuple[float, float] | None = None,
) -> list[AxisAlignedPrism]:
    if geometry_node is None:
        return []

    prisms: list[AxisAlignedPrism] = []
    def _at_robot_height(items: list[AxisAlignedPrism]) -> list[AxisAlignedPrism]:
        if robot_z_range is None:
            return items
        zlo, zhi = sorted((float(robot_z_range[0]), float(robot_z_range[1])))
        return [item for item in items if item.zmin <= zhi and item.zmax >= zlo]
    box = _find_child(geometry_node, 'box')
    if box is not None:
        size_node = _find_child(box, 'size')
        if size_node is not None and size_node.text:
            prisms.append(
                _parse_box_prism(
                    f'{model_name}/{link_name}:{element_name}',
                    pose,
                    size_node.text.strip(),
                )
            )
        return _at_robot_height(prisms)

    cylinder = _find_child(geometry_node, 'cylinder')
    if cylinder is not None:
        radius_node = _find_child(cylinder, 'radius')
        length_node = _find_child(cylinder, 'length')
        if radius_node is not None and radius_node.text and length_node is not None and length_node.text:
            prisms.append(
                _parse_cylinder_prism(
                    f'{model_name}/{link_name}:{element_name}',
                    pose,
                    radius_node.text.strip(),
                    length_node.text.strip(),
                )
            )
        return _at_robot_height(prisms)

    mesh = _find_child(geometry_node, 'mesh')
    if mesh is not None:
        uri_node = _find_child(mesh, 'uri')
        if uri_node is None or not uri_node.text:
            raise RuntimeError(f'Mesh collision {model_name}/{link_name}:{element_name} has no URI')
        scale_node = _find_child(mesh, 'scale')
        scale = (1.0, 1.0, 1.0)
        if scale_node is not None and scale_node.text:
            scale = tuple(float(v) for v in scale_node.text.split())
            if len(scale) != 3:
                raise RuntimeError(f'Mesh scale must have 3 values: {scale_node.text}')
        mesh_path = _resolve_model_uri(uri_node.text.strip(), model_search_paths)
        points = _collada_position_points(mesh_path, scale)
        points = _transform_points(points, pose)
        if robot_z_range is not None:
            points = _points_touching_z_slab(points, robot_z_range)
        if points.size:
            mins = points.min(axis=0)
            maxs = points.max(axis=0)
            prisms.append(AxisAlignedPrism(
                name=f'{model_name}/{link_name}:{element_name}',
                xmin=float(mins[0]), xmax=float(maxs[0]),
                ymin=float(mins[1]), ymax=float(maxs[1]),
                zmin=float(mins[2]), zmax=float(maxs[2]),
            ))
    return _at_robot_height(prisms)


def _resolve_model_uri(uri: str, search_paths: Sequence[str]) -> str:
    if not uri.startswith('model://'):
        path = Path(uri).expanduser()
        if path.is_file():
            return str(path.resolve())
        raise RuntimeError(f'Unsupported or missing mesh URI: {uri}')
    relative = uri[len('model://'):]
    for root in search_paths:
        candidate = Path(root) / relative
        if candidate.is_file():
            return str(candidate.resolve())
    raise RuntimeError(f'Could not resolve {uri} in model search paths: {list(search_paths)}')


def _collada_position_points(path: str, scale: Sequence[float]) -> np.ndarray:
    root = ET.parse(path).getroot()
    unit = 1.0
    for node in root.iter():
        if _tag_matches(node, 'unit') and node.get('meter'):
            unit = float(node.get('meter', '1'))
            break
    arrays = [node for node in root.iter() if _tag_matches(node, 'float_array')]
    position_arrays = [node for node in arrays if 'position' in node.get('id', '').lower()]
    if not position_arrays or not position_arrays[0].text:
        raise RuntimeError(f'COLLADA mesh has no POSITION array: {path}')
    values = np.asarray([float(v) for v in position_arrays[0].text.split()], dtype=float)
    if values.size % 3:
        raise RuntimeError(f'COLLADA POSITION array is not XYZ triples: {path}')
    return values.reshape(-1, 3) * (unit * np.asarray(scale, dtype=float))


def _transform_points(points: np.ndarray, pose: _Pose6D) -> np.ndarray:
    cr, sr = math.cos(pose.roll), math.sin(pose.roll)
    cp, sp = math.cos(pose.pitch), math.sin(pose.pitch)
    cy, sy = math.cos(pose.yaw), math.sin(pose.yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return points @ (rz @ ry @ rx).T + np.array([pose.x, pose.y, pose.z])


def _points_touching_z_slab(points: np.ndarray, z_range: tuple[float, float]) -> np.ndarray:
    """Return vertices in a slab plus edge intersections for a conservative slice AABB."""
    zmin, zmax = sorted((float(z_range[0]), float(z_range[1])))
    inside = points[(points[:, 2] >= zmin) & (points[:, 2] <= zmax)]
    extras = []
    # Without triangle indices, all-pairs would invent geometry. The in-slab
    # vertices are sufficient for the current collision meshes; include global
    # extrema only when the mesh spans the slab but has no sampled vertex in it.
    if inside.size == 0 and points[:, 2].min() <= zmax and points[:, 2].max() >= zmin:
        extras = [points]
    if extras:
        return np.concatenate([inside, *extras], axis=0)
    return inside


def _normalize_names(names: str | Sequence[str] | None) -> tuple[str, ...]:
    if names is None:
        return ()
    if isinstance(names, str):
        value = str(names).strip()
        return (value,) if value else ()
    out = []
    for item in names:
        value = str(item).strip()
        if value:
            out.append(value)
    return tuple(out)


def _normalize_geometry_tags(geometry_tags: Iterable[str] | None) -> tuple[str, ...]:
    if geometry_tags is None:
        return ('visual',)
    out = []
    for item in geometry_tags:
        value = str(item).strip().lower()
        if value:
            out.append(value)
    return tuple(out) if out else ('visual',)


def parse_occlusion_scene_from_world(
    world_path: str,
    *,
    model_name: str | Sequence[str] = 'warehouse_rack_occluders',
    geometry_tags: Iterable[str] | None = None,
    include_names: Sequence[str] = (),
    model_search_paths: Sequence[str] = (),
    robot_z_range: tuple[float, float] | None = None,
) -> OcclusionScene:
    if not os.path.isfile(world_path):
        raise RuntimeError(f'World file not found: {world_path}')
    try:
        tree = ET.parse(world_path)
    except ET.ParseError as exc:
        raise RuntimeError(f"Failed to parse world file '{world_path}': {exc}") from exc

    root = tree.getroot()
    default_models = str(Path(world_path).resolve().parents[2] / 'models')
    search_paths = tuple(model_search_paths) or (default_models,)
    model_names = _normalize_names(model_name)
    geometry_kinds = _normalize_geometry_tags(geometry_tags)
    matched_models: list[ET.Element] = []
    for node in root.iter():
        if not _tag_matches(node, 'model'):
            continue
        node_name = str(node.attrib.get('name', '')).strip()
        if node_name in model_names:
            matched_models.append(node)

    include_name_set = set(_normalize_names(include_names))
    world_node = next((node for node in root.iter() if _tag_matches(node, 'world')), None)
    if world_node is not None:
        for include in list(world_node):
            if not _tag_matches(include, 'include'):
                continue
            name_node = _find_child(include, 'name')
            instance_name = name_node.text.strip() if name_node is not None and name_node.text else ''
            if instance_name not in include_name_set:
                continue
            uri_node = _find_child(include, 'uri')
            if uri_node is None or not uri_node.text:
                raise RuntimeError(f'Included collision model {instance_name} has no URI')
            model_dir = _resolve_model_uri(uri_node.text.strip() + '/model.sdf', search_paths)
            included_root = ET.parse(model_dir).getroot()
            included_model = next(
                (node for node in included_root.iter() if _tag_matches(node, 'model')), None)
            if included_model is None:
                raise RuntimeError(f'Included model has no <model>: {model_dir}')
            included_model = ET.fromstring(ET.tostring(included_model, encoding='unicode'))
            included_model.set('name', instance_name)
            include_pose = _parse_pose_node(_find_child(include, 'pose'))
            model_pose = _parse_pose_node(_find_child(included_model, 'pose'))
            composed = _compose_pose(include_pose, model_pose)
            pose_node = _find_child(included_model, 'pose')
            if pose_node is None:
                pose_node = ET.Element('pose')
                included_model.insert(0, pose_node)
            pose_node.text = (
                f'{composed.x} {composed.y} {composed.z} '
                f'{composed.roll} {composed.pitch} {composed.yaw}'
            )
            matched_models.append(included_model)

    if not matched_models:
        return OcclusionScene(
            prisms=(),
            source_world=world_path,
            model_name=','.join(model_names) if model_names else '',
        )

    prisms: list[AxisAlignedPrism] = []
    for model_node in matched_models:
        node_model_name = str(model_node.attrib.get('name', 'model')).strip() or 'model'
        model_pose = _parse_pose_node(_find_child(model_node, 'pose'))
        for link in list(model_node):
            if not _tag_matches(link, 'link'):
                continue
            link_name = str(link.attrib.get('name', 'link'))
            link_pose = _compose_pose(model_pose, _parse_pose_node(_find_child(link, 'pose')))
            for element in list(link):
                if element.tag.split('}')[-1].lower() not in geometry_kinds:
                    continue
                element_kind = element.tag.split('}')[-1].lower()
                element_name = str(element.attrib.get('name', element_kind))
                element_pose = _compose_pose(link_pose, _parse_pose_node(_find_child(element, 'pose')))
                prisms.extend(
                    _geometry_prisms_from_node(
                        model_name=node_model_name,
                        link_name=link_name,
                        geometry_node=_find_child(element, 'geometry'),
                        pose=element_pose,
                        element_name=element_name,
                        model_search_paths=search_paths,
                        robot_z_range=robot_z_range,
                    )
                )
    return OcclusionScene(
        prisms=tuple(prisms),
        source_world=world_path,
        model_name=','.join(model_names) if model_names else '',
    )


def parse_collision_scene_from_world(
    world_path: str,
    *,
    model_names: Sequence[str] = ('warehouse_walls', 'warehouse_rack_occluders'),
    include_names: Sequence[str] = (),
    model_search_paths: Sequence[str] = (),
    robot_z_range: tuple[float, float] | None = (0.07, 0.35),
) -> OcclusionScene:
    return parse_occlusion_scene_from_world(
        world_path,
        model_name=model_names,
        geometry_tags=('collision',),
        include_names=include_names,
        model_search_paths=model_search_paths,
        robot_z_range=robot_z_range,
    )


def profile_collision_scene(
    world_path: str,
    profile: dict,
    *,
    robot_z_range: tuple[float, float] | None = (0.0, 0.55),
) -> OcclusionScene:
    """The complete collision scene of a world, named by its world profile.

    Takes both the model groups (`collision_model_names`) and the top-level included
    objects (`collision_include_names`); omitting the latter silently drops loose objects
    such as a parked forklift. Raises if the scene is empty, which is what a name that
    matches nothing would otherwise produce.
    """
    names = tuple(profile.get("collision_model_names") or ())
    if not names:
        raise ValueError(f"world profile for {world_path} declares no collision_model_names")
    scene = parse_collision_scene_from_world(
        world_path, model_names=names,
        include_names=tuple(profile.get("collision_include_names") or ()),
        robot_z_range=robot_z_range)
    if not scene.prisms:
        raise ValueError(f"collision_model_names {names} matched no geometry in {world_path}")
    return scene


def scene_from_json(text: str | None) -> OcclusionScene:
    return OcclusionScene.from_json(text)


def scene_to_json(scene: OcclusionScene) -> str:
    return scene.to_json()


def top_heights_for_xy(prisms: tuple[AxisAlignedPrism, ...] | list[AxisAlignedPrism], xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(xy, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, 2)
    heights = np.full(pts.shape[0], _EMPTY_HEIGHT, dtype=float)
    for prism in prisms:
        mask = (
            (pts[:, 0] >= prism.xmin)
            & (pts[:, 0] <= prism.xmax)
            & (pts[:, 1] >= prism.ymin)
            & (pts[:, 1] <= prism.ymax)
        )
        if np.any(mask):
            heights[mask] = np.maximum(heights[mask], prism.zmax)
    return heights


def top_height_at_xy(prisms: tuple[AxisAlignedPrism, ...] | list[AxisAlignedPrism], x: float, y: float) -> float:
    return float(top_heights_for_xy(prisms, np.array([[x, y]], dtype=float))[0])


import functools

@functools.lru_cache(maxsize=16)
def _get_union_boundary_segments(prisms: tuple[AxisAlignedPrism, ...]) -> list[tuple[np.ndarray, np.ndarray]]:
    if not prisms:
        return []
    
    prisms_list = list(prisms)

    def covered_outside(px, py, ignore_idx):
        """True if the point (just outside one prism's edge) lies inside ANY other prism.
        Used to drop INTERNAL union seams: an edge of prism P is part of the true union
        boundary only if the space immediately on its OUTWARD side is free. This correctly
        dissolves both overlapping prisms AND exactly-abutting prisms (whose shared edge the
        old midpoint-interior test missed, leaving phantom internal boundary segments)."""
        for idx2, q in enumerate(prisms_list):
            if idx2 == ignore_idx:
                continue
            if (q.xmin - 1e-6 <= px <= q.xmax + 1e-6) and (q.ymin - 1e-6 <= py <= q.ymax + 1e-6):
                return True
        return False

    probe = 1e-3  # outward offset (smaller than any lane width, larger than float noise)
    boundary_segments = []
    for idx, p in enumerate(prisms_list):
        # each edge carries its OUTWARD normal (pointing away from this prism's interior)
        edges = [
            (np.array([p.xmin, p.ymin]), np.array([p.xmin, p.ymax]), 'vertical', np.array([-1.0, 0.0])),
            (np.array([p.xmax, p.ymin]), np.array([p.xmax, p.ymax]), 'vertical', np.array([1.0, 0.0])),
            (np.array([p.xmin, p.ymin]), np.array([p.xmax, p.ymin]), 'horizontal', np.array([0.0, -1.0])),
            (np.array([p.xmin, p.ymax]), np.array([p.xmax, p.ymax]), 'horizontal', np.array([0.0, 1.0])),
        ]
        for p1, p2, orient, outward in edges:
            splits = [0.0, 1.0]
            if orient == 'vertical':
                x_val = p1[0]
                y_start, y_end = p1[1], p2[1]
                for o_idx, op in enumerate(prisms_list):
                    if o_idx == idx:
                        continue
                    if op.xmin - 1e-5 < x_val < op.xmax + 1e-5:
                        y_min_clip = max(y_start, op.ymin)
                        y_max_clip = min(y_end, op.ymax)
                        if y_min_clip < y_max_clip:
                            splits.append((y_min_clip - y_start) / (y_end - y_start))
                            splits.append((y_max_clip - y_start) / (y_end - y_start))
            else:
                y_val = p1[1]
                x_start, x_end = p1[0], p2[0]
                for o_idx, op in enumerate(prisms_list):
                    if o_idx == idx:
                        continue
                    if op.ymin - 1e-5 < y_val < op.ymax + 1e-5:
                        x_min_clip = max(x_start, op.xmin)
                        x_max_clip = min(x_end, op.xmax)
                        if x_min_clip < x_max_clip:
                            splits.append((x_min_clip - x_start) / (x_end - x_start))
                            splits.append((x_max_clip - x_start) / (x_end - x_start))

            splits = sorted(list(set([float(np.clip(s, 0.0, 1.0)) for s in splits])))
            for i in range(len(splits) - 1):
                s1, s2 = splits[i], splits[i+1]
                if s2 - s1 < 1e-5:
                    continue
                smid = 0.5 * (s1 + s2)
                pt_mid = p1 + smid * (p2 - p1)
                # keep only TRUE boundary: the outward side of this sub-edge is free space
                # (not covered by an overlapping OR abutting prism)
                if not covered_outside(pt_mid[0] + probe * outward[0],
                                       pt_mid[1] + probe * outward[1], idx):
                    boundary_segments.append((
                        p1 + s1 * (p2 - p1),
                        p1 + s2 * (p2 - p1)
                    ))
    return boundary_segments


def signed_distance_to_union_xy(
    prisms: tuple[AxisAlignedPrism, ...] | list[AxisAlignedPrism],
    xy: np.ndarray,
    *,
    keep_in: bool = True,
) -> np.ndarray:
    pts = np.asarray(xy, dtype=float)
    is_1d = (pts.ndim == 1)
    if is_1d:
        pts = pts.reshape(1, 2)
    if not prisms:
        return np.full(pts.shape[0], np.inf, dtype=float)
    
    if not keep_in:
        signed = np.full(pts.shape[0], np.inf, dtype=float)
        for prism in prisms:
            signed = np.minimum(signed, prism.signed_distance_xy(pts))
        return signed
        
    segs = _get_union_boundary_segments(tuple(prisms))
    if not segs:
        return np.full(pts.shape[0], np.inf, dtype=float)
        
    n_pts = pts.shape[0]
    min_dists = np.full(n_pts, np.inf, dtype=float)
    
    for p1, p2 in segs:
        v = p2 - p1
        v_len_sq = np.sum(v**2)
        if v_len_sq < 1e-9:
            d = np.linalg.norm(pts - p1, axis=1)
        else:
            w = pts - p1
            t = np.clip(np.dot(w, v) / v_len_sq, 0.0, 1.0)
            closest = p1 + t[:, np.newaxis] * v
            d = np.linalg.norm(pts - closest, axis=1)
        min_dists = np.minimum(min_dists, d)
        
    is_inside = np.zeros(n_pts, dtype=bool)
    for p in prisms:
        dx = np.maximum(np.maximum(p.xmin - pts[:, 0], 0.0), pts[:, 0] - p.xmax)
        dy = np.maximum(np.maximum(p.ymin - pts[:, 1], 0.0), pts[:, 1] - p.ymax)
        is_inside |= (dx <= 0.0) & (dy <= 0.0)
        
    signed = np.where(is_inside, -min_dists, min_dists)
    return signed



def segment_intersects_prism(start_xyz: np.ndarray, end_xyz: np.ndarray, prism: AxisAlignedPrism, *, eps: float = 1e-9) -> bool:
    p0 = np.asarray(start_xyz, dtype=float)
    p1 = np.asarray(end_xyz, dtype=float)
    direction = p1 - p0
    bounds_min = np.array([prism.xmin, prism.ymin, prism.zmin], dtype=float)
    bounds_max = np.array([prism.xmax, prism.ymax, prism.zmax], dtype=float)

    t_min = 0.0
    t_max = 1.0
    for axis in range(3):
        if abs(direction[axis]) <= eps:
            if p0[axis] < bounds_min[axis] or p0[axis] > bounds_max[axis]:
                return False
            continue
        inv = 1.0 / direction[axis]
        t0 = (bounds_min[axis] - p0[axis]) * inv
        t1 = (bounds_max[axis] - p0[axis]) * inv
        if t0 > t1:
            t0, t1 = t1, t0
        t_min = max(t_min, t0)
        t_max = min(t_max, t1)
        if t_max < t_min:
            return False
    return (t_max >= max(t_min, 0.0)) and (t_min <= 1.0)


def segment_occluded(
    prisms: tuple[AxisAlignedPrism, ...] | list[AxisAlignedPrism],
    start_xyz: np.ndarray,
    end_xyz: np.ndarray,
) -> bool:
    return any(segment_intersects_prism(start_xyz, end_xyz, prism) for prism in prisms)
