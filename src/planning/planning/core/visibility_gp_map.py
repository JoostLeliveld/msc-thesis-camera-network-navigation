"""Empirical GP visibility-map model loaded from an offline artifact."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
from pathlib import Path

import numpy as np
import scipy.interpolate
from planning.core.plan_validation import immutable_array


def _clip_prob(p, eps: float):
    return np.clip(p, eps, 1.0 - eps)


@dataclass
class GPVisibilityMapConfig:
    artifact_path: str = ""
    camera_pos: tuple[float, float, float] = (-3.0, -3.0, 6.0)
    target_height_m: float = 0.0
    min_prob: float = 1e-4


class GPVisibilityMapModel:
    """Load a fixed GP visibility field used by the thesis experiments."""

    def __setattr__(self, name, value):
        if getattr(self, '_loaded', False) and not name.startswith('_'):
            raise AttributeError('loaded visibility field is immutable; construct a new model')
        object.__setattr__(self, name, value)

    def __init__(self, cfg: GPVisibilityMapConfig):
        self.cfg = cfg
        self.min_prob = float(max(cfg.min_prob, 1e-6))
        if not np.isfinite(cfg.min_prob) or not 0 < cfg.min_prob < .5:
            raise ValueError('visibility min_prob must be finite and in (0,.5)')
        self.artifact_path = self._resolve_artifact_path(cfg.artifact_path)
        self._load_artifact()
        self._prob_state_casadi = None
        self._loaded = True

    @staticmethod
    def _resolve_artifact_path(raw_path: str) -> Path:
        path = Path(str(raw_path or "").strip()).expanduser()
        if not str(path):
            raise RuntimeError(
                "GP visibility planning requires 'visibility_artifact_path' to point to an empirical GP artifact."
            )
        if not path.is_file():
            raise RuntimeError(f"Empirical GP visibility artifact not found: {path}")
        return path.resolve()

    def _load_artifact(self) -> None:
        artifact_bytes = self.artifact_path.read_bytes()
        self.sha256 = hashlib.sha256(artifact_bytes).hexdigest()
        try:
            with np.load(io.BytesIO(artifact_bytes), allow_pickle=False) as data:
                xs = np.asarray(data["xs"], dtype=float)
                ys = np.asarray(data["ys"], dtype=float)
                if "P_conservative_plan_map" not in data.files:
                    raise RuntimeError(
                        f"Empirical GP visibility artifact {self.artifact_path} uses an outdated schema "
                        "(missing P_conservative_plan_map). Re-fit the visibility GPs."
                    )
                p_cons = np.asarray(data["P_conservative_plan_map"], dtype=float)
                p_mean = np.asarray(data["P_mean_map"], dtype=float)
                camera_pos = (
                    np.asarray(data["camera_pos"], dtype=float).reshape(-1)
                    if "camera_pos" in data.files
                    else np.asarray(self.cfg.camera_pos, dtype=float).reshape(-1)
                )
                target_height = (
                    float(np.asarray(data["target_height"], dtype=float).reshape(-1)[0])
                    if "target_height" in data.files
                    else float(self.cfg.target_height_m)
                )
        except KeyError as exc:
            raise RuntimeError(
                f"Empirical GP visibility artifact {self.artifact_path} is missing required field {exc!s}."
            ) from exc

        if xs.ndim != 1 or ys.ndim != 1 or xs.size < 2 or ys.size < 2:
            raise RuntimeError(
                f"Empirical GP visibility artifact {self.artifact_path} has an invalid grid."
            )
        expected_shape = (ys.size, xs.size)
        if any(not np.isfinite(axis).all() or not (np.diff(axis)>0).all() for axis in (xs,ys)):
            raise ValueError('visibility axes must be finite and strictly increasing')
        for field_name, grid in (
            ("P_mean_map", p_mean),
            ("P_conservative_plan_map", p_cons),
        ):
            if grid.shape != expected_shape:
                raise RuntimeError(
                    f"Empirical GP visibility artifact {self.artifact_path} has {field_name} shape {grid.shape}, "
                    f"expected {expected_shape}."
                )
            if not np.isfinite(grid).all() or np.any((grid < 0) | (grid > 1)):
                raise ValueError(f'{field_name} must be finite probabilities in [0,1]')
        if camera_pos.shape != (3,) or not np.isfinite(camera_pos).all() or not np.isfinite(target_height):
            raise ValueError('visibility camera and target geometry must be finite')

        self.xs = immutable_array(xs)
        self.ys = immutable_array(ys)
        self.P_mean_map = immutable_array(_clip_prob(p_mean, self.min_prob))
        self.P_conservative_plan_map = immutable_array(_clip_prob(p_cons, self.min_prob))
        self.camera_pos = immutable_array(camera_pos)
        self.target_height = float(target_height)
        self.x_min = float(self.xs[0])
        self.x_max = float(self.xs[-1])
        self.y_min = float(self.ys[0])
        self.y_max = float(self.ys[-1])
        self._prob_interp = scipy.interpolate.RegularGridInterpolator(
            (self.ys, self.xs), 
            self.P_conservative_plan_map, 
            method='linear', 
            bounds_error=True,
            fill_value=None,
        )

    def contains_xy_np(self, x: float, y: float) -> bool:
        if not (np.isfinite(x) and np.isfinite(y)):
            return False
        return bool(
            self.x_min <= float(x) <= self.x_max
            and self.y_min <= float(y) <= self.y_max
        )

    def _require_xy_in_support_np(self, x: float, y: float) -> tuple[float, float]:
        x = float(x)
        y = float(y)
        if not self.contains_xy_np(x, y):
            raise RuntimeError(
                "GP visibility query outside artifact support: "
                f"x={x:.3f}, y={y:.3f}, "
                f"support=[{self.x_min:.3f},{self.x_max:.3f}]x"
                f"[{self.y_min:.3f},{self.y_max:.3f}]."
            )
        return x, y

    @property
    def signature(self) -> tuple:
        return (
            "empirical_gp_visibility",
            self.sha256, self.min_prob, tuple(self.camera_pos), self.target_height,
        )

    def prob_state_np(self, m) -> float:
        if len(m) < 2:
            raise ValueError(f"State vector m must have length >= 2, got {len(m)}")
        x, y = self._require_xy_in_support_np(float(m[0]), float(m[1]))
        p = self._prob_interp((y, x))
        return float(_clip_prob(p, self.min_prob))

    def make_prob_state_casadi(self):
        try:
            import casadi as ca
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("CasADi is not available for visibility model") from exc

        if self._prob_state_casadi is not None:
            return self._prob_state_casadi

        values = np.asarray(self.P_conservative_plan_map.T, dtype=float).ravel(order="F")
        interp = ca.interpolant(
            f"empirical_gp_visibility_{hashlib.sha1(str(self.artifact_path).encode('utf-8')).hexdigest()[:10]}",
            "linear",
            [self.xs.tolist(), self.ys.tolist()],
            values,
        )
        eps = float(self.min_prob)
        x_min = float(self.x_min)
        x_max = float(self.x_max)
        y_min = float(self.y_min)
        y_max = float(self.y_max)

        def p_vis_ca(m):
            inside = ca.logic_and(ca.logic_and(m[0] >= x_min, m[0] <= x_max),
                                  ca.logic_and(m[1] >= y_min, m[1] <= y_max))
            x = ca.fmin(ca.fmax(m[0], x_min), x_max)
            y = ca.fmin(ca.fmax(m[1], y_min), y_max)
            z = interp(ca.vertcat(x, y))
            return ca.if_else(inside, ca.fmin(ca.fmax(z, eps), 1.0 - eps), eps)

        self._prob_state_casadi = p_vis_ca
        return p_vis_ca
