#!/usr/bin/env python3
"""Measured inventory of the AWS RoboMaker warehouse mesh pack.

Nothing here is hand-typed: every footprint is the axis-aligned bounding box of
the *visual* DAE after applying the COLLADA <unit meter> factor and the scene
node transforms, and every colour is the mean of the model's own texture PNGs.
That makes the sketches drawn from this module dimensionally honest -- a
rectangle on the plan is the rectangle the camera will actually see.

Validation: ShelfD_01 measures 3.917 x 0.880 x 2.613 m here, which is exactly the
constant `MESH_L, MESH_W, MESH_H` that make_warehouse_full.py measured
independently. Same answer from a different parser.
"""
from __future__ import annotations

import json
import pathlib
import xml.etree.ElementTree as ET

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
MODELS = REPO / "src/sim/models"
CACHE = pathlib.Path(__file__).resolve().parent / "mesh_inventory.json"

C = "{http://www.collada.org/2005/11/COLLADASchema}"


def _q(tag: str) -> str:
    return f"{C}{tag}"


def _measure_dae(path: pathlib.Path):
    root = ET.parse(path).getroot()
    unit, up = 1.0, "Y_UP"
    asset = root.find(_q("asset"))
    if asset is not None:
        u = asset.find(_q("unit"))
        if u is not None and u.get("meter"):
            unit = float(u.get("meter"))
        a = asset.find(_q("up_axis"))
        if a is not None and a.text:
            up = a.text.strip()

    geo: dict[str, np.ndarray] = {}
    for g in root.iter(_q("geometry")):
        mesh = g.find(_q("mesh"))
        if mesh is None:
            continue
        srcs = {}
        for s in mesh.findall(_q("source")):
            fa = s.find(_q("float_array"))
            if fa is None or not fa.text:
                continue
            acc = s.find(f"{_q('technique_common')}/{_q('accessor')}")
            stride = int(acc.get("stride")) if acc is not None and acc.get("stride") else 3
            vals = np.fromstring(fa.text, sep=" ")
            if stride >= 3 and vals.size >= stride:
                srcs[s.get("id")] = vals.reshape(-1, stride)[:, :3]
        pts = []
        for v in mesh.findall(_q("vertices")):
            for inp in v.findall(_q("input")):
                if inp.get("semantic") == "POSITION":
                    sid = inp.get("source").lstrip("#")
                    if sid in srcs:
                        pts.append(srcs[sid])
        if pts:
            geo[g.get("id")] = np.vstack(pts)

    chunks: list[np.ndarray] = []

    def walk(node, M):
        L = M.copy()
        for ch in node:
            if ch.tag == _q("matrix"):
                L = L @ np.fromstring(ch.text, sep=" ").reshape(4, 4)
            elif ch.tag == _q("translate"):
                T = np.eye(4); T[:3, 3] = np.fromstring(ch.text, sep=" "); L = L @ T
            elif ch.tag == _q("scale"):
                S = np.eye(4); S[:3, :3] = np.diag(np.fromstring(ch.text, sep=" ")); L = L @ S
            elif ch.tag == _q("rotate"):
                v = np.fromstring(ch.text, sep=" ")
                ax, ang = v[:3], np.deg2rad(v[3])
                n = np.linalg.norm(ax)
                if n > 0:
                    ax = ax / n
                    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
                    R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
                    T = np.eye(4); T[:3, :3] = R; L = L @ T
        for ig in node.findall(_q("instance_geometry")):
            gid = ig.get("url").lstrip("#")
            if gid in geo:
                P = geo[gid]
                chunks.append((np.hstack([P, np.ones((len(P), 1))]) @ L.T)[:, :3])
        for ch in node.findall(_q("node")):
            walk(ch, L)

    lib = root.find(_q("library_visual_scenes"))
    if lib is not None:
        for vs in lib.findall(_q("visual_scene")):
            for n in vs.findall(_q("node")):
                walk(n, np.eye(4))
    if not chunks:
        chunks = list(geo.values())
    if not chunks:
        return None

    P = np.vstack(chunks) * unit
    if up.startswith("Y"):
        P = np.column_stack([P[:, 0], -P[:, 2], P[:, 1]])
    elif up.startswith("X"):
        P = np.column_stack([-P[:, 1], P[:, 0], P[:, 2]])
    return P.min(0), P.max(0)


def _texture_colour(model_dir: pathlib.Path):
    """Mean RGB of the model's textures, so plan colours match the rendered look."""
    try:
        from PIL import Image
    except Exception:
        return None
    pngs = sorted((model_dir / "materials/textures").glob("*.png"))
    if not pngs:
        return None
    acc, wsum = np.zeros(3), 0.0
    for p in pngs:
        try:
            im = Image.open(p).convert("RGB").resize((64, 64))
        except Exception:
            continue
        a = np.asarray(im, dtype=float).reshape(-1, 3) / 255.0
        # drop near-black and near-white pixels: those are shadow bake and
        # unpainted background, not the colour a person would call the object's.
        lum = a.mean(1)
        keep = (lum > 0.10) & (lum < 0.96)
        if keep.sum() < 32:
            keep = np.ones(len(a), bool)
        acc += a[keep].mean(0) * keep.sum()
        wsum += keep.sum()
    if wsum == 0:
        return None
    return [round(float(v), 4) for v in acc / wsum]


def build(force: bool = False) -> dict:
    if CACHE.exists() and not force:
        return json.loads(CACHE.read_text())
    out = {}
    for d in sorted(MODELS.glob("aws_robomaker_warehouse_*")):
        vis = sorted(d.glob("meshes/*_visual.DAE"))
        if not vis:
            continue
        r = _measure_dae(vis[0])
        if r is None:
            continue
        lo, hi = r
        short = d.name.replace("aws_robomaker_warehouse_", "")
        out[short] = {
            "model": d.name,
            "size": [round(float(v), 4) for v in (hi - lo)],
            "zmin": round(float(lo[2]), 4),
            "zmax": round(float(hi[2]), 4),
            "centre_xy": [round(float(0.5 * (lo[i] + hi[i])), 4) for i in (0, 1)],
            "colour": _texture_colour(d),
            "n_textures": len(list((d / "materials/textures").glob("*.png"))),
        }
    CACHE.write_text(json.dumps(out, indent=2) + "\n")
    return out


MESHES = build()


def footprint(name: str) -> tuple[float, float]:
    m = MESHES[name]
    return m["size"][0], m["size"][1]


def height(name: str) -> float:
    return MESHES[name]["size"][2]


def colour(name: str, fallback="#9aa5ad") -> str:
    c = MESHES[name].get("colour")
    if not c:
        return fallback
    return "#%02x%02x%02x" % tuple(int(round(255 * v)) for v in c)


if __name__ == "__main__":
    inv = build(force=True)
    hdr = f"{'mesh':<18}{'X':>7}{'Y':>7}{'Z':>7}{'zmin':>7}{'zmax':>7}  {'colour':<9}{'tex':>4}"
    print(hdr); print("-" * len(hdr))
    for k, v in inv.items():
        sx, sy, sz = v["size"]
        print(f"{k:<18}{sx:7.2f}{sy:7.2f}{sz:7.2f}{v['zmin']:7.2f}{v['zmax']:7.2f}  {colour(k):<9}{v['n_textures']:4d}")
