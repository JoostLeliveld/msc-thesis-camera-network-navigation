"""Smooth no-go-zone obstacle penalties derived from warehouse prism geometry."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from unav_common.occlusion_geometry import scene_from_json, signed_distance_to_union_xy, _get_union_boundary_segments
from unav_common.navigation_parameters import validate_navigation_parameters
from unav_common.rectangular_footprint import RectangularFootprint


VALID_NOGO_PENALTIES = ('warning_band',)


@dataclass(frozen=True)
class NogoCostConfig:
    penalty_type: str = 'warning_band'
    safe_distance: float = 0.35
    logbarrier_eps: float = 1e-3
    # warning_band: hinged-log warning penalty parameters (penalty_type ==
    # 'warning_band'). `warning_band` (b) is the clearance width over which the
    # soft warning ramps in; `near_weight` (w_near) scales the warning term. The
    # violation term reuses `weight` (w_viol) and `logbarrier_eps` (eps). The
    # penalty is exactly zero for valid interior states with clearance >= b, so
    # raising `weight` to crush violations does not bias the choice between two
    # fully valid routes (e.g. a narrow vs a wide aisle).
    warning_band: float = 0.05
    near_weight: float = 50.0
    # Rectangular body half-extents. When both are set the clearance uses the
    # heading-aware support distance a|cos|+b|sin| instead of the fixed
    # ``safe_distance`` disc, so the cost distinguishes driving aligned with an
    # aisle from crossing it at an angle. ``body_margin`` is the keep-clear
    # margin added on top of the body itself. The warning band is separate: a
    # zero body margin with warning_band=0.05 means the cost starts exactly
    # when the oriented body comes within 5 cm of geometry.
    robot_half_length: float = None
    robot_half_width: float = None
    body_margin: float = 0.0
    # How much steeper contact is than proximity in the single clearance
    # penalty. Sets the ratio between the two regimes, not an absolute scale.
    contact_gain: float = 100.0
    geometry_json: str = ''
    # 'keep_out': penalise being inside/near the prisms (obstacle footprints).
    # 'keep_in':  penalise leaving the prism union (driveable region).
    mode: str = 'keep_out'


class NogoZoneCostModel:
    """Geometry-based no-go-zone penalty around obstacle footprints."""

    def __init__(self, cfg: NogoCostConfig):
        validate_navigation_parameters({
            'nogo_safe_distance': cfg.safe_distance,
            'nogo_logbarrier_eps': cfg.logbarrier_eps,
            'nogo_warning_band': cfg.warning_band, 'nogo_near_weight': cfg.near_weight,
        })
        penalty_type = str(cfg.penalty_type or '').strip().lower()
        if penalty_type not in VALID_NOGO_PENALTIES:
            raise ValueError(
                f"penalty_type must be one of: {', '.join(VALID_NOGO_PENALTIES)}"
            )
        self.cfg = cfg
        self.penalty_type = penalty_type
        self.mode = str(getattr(cfg, 'mode', 'keep_out') or 'keep_out').strip().lower()
        if self.mode not in ('keep_out', 'keep_in'):
            raise ValueError("mode must be 'keep_out' or 'keep_in'")
        self.safe_distance = float(max(cfg.safe_distance, 0.0))
        self.logbarrier_eps = float(max(cfg.logbarrier_eps, 1e-6))
        self.warning_band = float(max(getattr(cfg, 'warning_band', 0.05), 1e-6))
        self.near_weight = float(max(getattr(cfg, 'near_weight', 50.0), 0.0))
        _half_l = getattr(cfg, 'robot_half_length', None)
        _half_w = getattr(cfg, 'robot_half_width', None)
        self.robot_half_length = None if _half_l is None else float(_half_l)
        self.robot_half_width = None if _half_w is None else float(_half_w)
        self.body_margin = float(max(getattr(cfg, 'body_margin', 0.0), 0.0))
        self.contact_gain = float(max(getattr(cfg, 'contact_gain', 100.0), 0.0))

        self.scene = scene_from_json(cfg.geometry_json)
        self.prisms = tuple(self.scene.prisms)

        self._xmins = np.asarray([float(p.xmin) for p in self.prisms], dtype=float)
        self._xmaxs = np.asarray([float(p.xmax) for p in self.prisms], dtype=float)
        self._ymins = np.asarray([float(p.ymin) for p in self.prisms], dtype=float)
        self._ymaxs = np.asarray([float(p.ymax) for p in self.prisms], dtype=float)
        self.union_boundary_segments = _get_union_boundary_segments(self.prisms)
        self._body_footprint = None
        if self.robot_half_length is not None and self.robot_half_width is not None:
            self._body_footprint = RectangularFootprint(
                self.prisms,
                length=2.0 * self.robot_half_length,
                width=2.0 * self.robot_half_width,
                keep_in=self.mode == 'keep_in',
            )

    @property
    def enabled(self) -> bool:
        return bool(self.prisms)

    @property
    def signature(self) -> tuple:
        scene_sig = []
        for prism in self.prisms:
            scene_sig.extend([
                round(float(prism.xmin), 4),
                round(float(prism.xmax), 4),
                round(float(prism.ymin), 4),
                round(float(prism.ymax), 4),
                round(float(prism.zmin), 4),
                round(float(prism.zmax), 4),
            ])
        return (
            'nogo_cost',
            self.mode,
            self.penalty_type,
            round(self.safe_distance, 6),
            round(self.logbarrier_eps, 8),
            round(self.warning_band, 6),
            round(self.near_weight, 6),
            None if self.robot_half_length is None else round(self.robot_half_length, 6),
            None if self.robot_half_width is None else round(self.robot_half_width, 6),
            round(self.body_margin, 6),
            len(self.prisms),
            *scene_sig,
        )

    def support_distance(self, yaw) -> float:
        """Half-extent of the rectangular body along the nearest wall normal.

        The planner cost previously inflated the centre point by a FIXED disc,
        so it could not distinguish driving aligned with an aisle (which needs
        the half-width) from crossing it at an angle (which needs up to the
        circumscribed radius). For a rectangle of half-length a and half-width
        b at heading ``yaw`` relative to the wall normal, the support distance
        is ``a|cos yaw| + b|sin yaw|``: 0.275 m aligned, 0.400 m facing the
        wall, and a maximum of 0.485 m at the diagonal - which is exactly the
        heading a robot passes through while turning. Lane rectangles are
        axis-aligned, so the nearest normal is a coordinate axis and the
        expression needs only the heading.
        """
        if self.robot_half_length is None or self.robot_half_width is None:
            return float(self.safe_distance)
        a = float(self.robot_half_length); b = float(self.robot_half_width)
        c = abs(math.cos(float(yaw))); s = abs(math.sin(float(yaw)))
        # Take the worse of the two axis normals so the cost is conservative
        # whichever wall is nearest.
        return max(a * c + b * s, a * s + b * c) + float(self.body_margin)

    def _clearance_np(self, xy: np.ndarray, yaw=None) -> float:
        if self._body_footprint is not None and yaw is not None:
            # Use the actual oriented body against the union geometry.  The old
            # surrogate took the larger of the x/y support distances without
            # identifying which boundary was nearest.  In a narrow straight
            # aisle that charged the longitudinal half-length against a lateral
            # wall and could label a physically clear, aligned route as an
            # overlap.  The exact footprint is already the authoritative hard
            # geometry model, so reuse it for the numerical soft cost and add
            # only the separately configured keep-clear margin.
            pose = np.array([float(xy[0]), float(xy[1]), float(yaw)], dtype=float)
            return float(self._body_footprint.clearance(pose) - self.body_margin)
        keep_in_flag = (self.mode == 'keep_in')
        signed_d = float(signed_distance_to_union_xy(self.prisms, np.asarray(xy, dtype=float), keep_in=keep_in_flag)[0])
        required = self.safe_distance if yaw is None else self.support_distance(yaw)
        if self.mode == 'keep_in':
            # signed_d <= 0 inside the driveable union. Positive clearance
            # means the mean state is safely inside the known driveable floor.
            return -signed_d - required
        return signed_d - required

    def signed_distance_state_np(self, m) -> float:
        if not self.prisms:
            return float('inf')
        xy = np.array([float(m[0]), float(m[1])], dtype=float)
        keep_in_flag = (self.mode == 'keep_in')
        signed_d = float(signed_distance_to_union_xy(self.prisms, xy, keep_in=keep_in_flag)[0])
        if self.mode == 'keep_in':
            # Positive means inside the known driveable union; negative means
            # outside it. This keeps penetration/inside diagnostics aligned with
            # "violation depth" instead of reporting valid lanes as no-go.
            return -signed_d
        return signed_d

    def penetration_depth_state_np(self, m) -> float:
        signed_d = self.signed_distance_state_np(m)
        return float(max(-signed_d, 0.0)) if np.isfinite(signed_d) else 0.0

    def clearance_state_np(self, m) -> float:
        if not self.enabled:
            return float('inf')
        xy = np.array([float(m[0]), float(m[1])], dtype=float)
        yaw = float(m[2]) if len(m) > 2 else None
        return float(self._clearance_np(xy, yaw))

    def inside_state_np(self, m) -> bool:
        return bool(self.penetration_depth_state_np(m) > 0.0)

    def _penalty_from_clearance_np(self, clearance: float) -> float:
        if not self.enabled:
            return 0.0

        # ONE continuous penalty in the clearance deficit, measured in units of
        # the warning band:
        #
        #     d = (warning_band - clearance) / warning_band
        #
        # d is 0 at the band edge, 1 where the body just touches the boundary,
        # and grows beyond 1 as it overlaps. The penalty is
        #
        #     near_weight * ( d^2  +  contact_gain * max(d - 1, 0)^2 )
        #
        # The first part penalises being CLOSE and rises smoothly across the
        # band. The second adds nothing until contact and then dominates, so an
        # overlap is categorically worse than proximity rather than merely
        # larger. Both pieces are quadratic, so the total is continuous and C1
        # at contact, and is exactly zero for clearance >= warning_band - two
        # routes that both keep clear are never separated by this term.
        #
        # This replaces an earlier hinged-log warning plus a separate quadratic
        # violation, which overlapped below zero clearance and made the shape
        # depend on two scales at once.
        deficit = max(self.warning_band - clearance, 0.0) / self.warning_band
        overlap = max(deficit - 1.0, 0.0)
        return float(self.near_weight * (deficit * deficit
                                         + self.contact_gain * overlap * overlap))

    def penalty_state_np(self, m) -> float:
        if not self.enabled:
            return 0.0
        xy = np.array([float(m[0]), float(m[1])], dtype=float)
        yaw = float(m[2]) if len(m) > 2 else None
        clearance = self._clearance_np(xy, yaw)
        return self._penalty_from_clearance_np(clearance)

    def penalty_belief_np(self, m, S, *, kappa: float = 1.0) -> float:
        """Expected no-go penalty under the current xy belief covariance."""
        if not self.enabled:
            return 0.0
        mean_xy = np.asarray([float(m[0]), float(m[1])], dtype=float)
        cov_xy = np.asarray(S, dtype=float)[:2, :2]
        cov_xy = 0.5 * (cov_xy + cov_xy.T)
        kappa = max(float(kappa), 1e-6)
        try:
            chol = np.linalg.cholesky(cov_xy + 1e-9 * np.eye(2))
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(cov_xy)
            chol = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 1e-9)))
        spread = math.sqrt(2.0 + kappa) * chol
        sigma_points = (
            mean_xy,
            mean_xy + spread[:, 0],
            mean_xy - spread[:, 0],
            mean_xy + spread[:, 1],
            mean_xy - spread[:, 1],
        )
        weights = (
            kappa / (2.0 + kappa),
            1.0 / (2.0 * (2.0 + kappa)),
            1.0 / (2.0 * (2.0 + kappa)),
            1.0 / (2.0 * (2.0 + kappa)),
            1.0 / (2.0 * (2.0 + kappa)),
        )
        yaw = float(m[2]) if len(m) > 2 else 0.0
        return float(sum(
            w * self.penalty_state_np([p[0], p[1], yaw])
            for p, w in zip(sigma_points, weights)
        ))

    def make_penalty_state_casadi(self):
        try:
            import casadi as ca
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError('CasADi is not available for no-go-zone cost') from exc

        if not self.prisms:
            def zero_penalty(_m):
                return 0.0
            return zero_penalty

        xmins, xmaxs = ca.DM(self._xmins), ca.DM(self._xmaxs)
        ymins, ymaxs = ca.DM(self._ymins), ca.DM(self._ymaxs)
        warning_band = float(self.warning_band)
        near_weight = float(self.near_weight)
        contact_gain = float(self.contact_gain)

        def body_clearance(m):
            if self.robot_half_length is None or self.robot_half_width is None:
                # Compatibility path for point/disc callers.
                dx = ca.fmax(ca.fmax(xmins - m[0], 0.0), m[0] - xmaxs)
                dy = ca.fmax(ca.fmax(ymins - m[1], 0.0), m[1] - ymaxs)
                outside = ca.sqrt(ca.power(dx, 2) + ca.power(dy, 2))
                inside_depth = ca.fmin(ca.fmin(m[0] - xmins, xmaxs - m[0]),
                                       ca.fmin(m[1] - ymins, ymaxs - m[1]))
                signed = ca.mmin(ca.if_else(
                    ca.logic_and(dx <= 0.0, dy <= 0.0), -inside_depth, outside))
                return (-signed if self.mode == 'keep_in' else signed) - self.safe_distance
            c, s = ca.fabs(ca.cos(m[2])), ca.fabs(ca.sin(m[2]))
            sx = self.robot_half_length * c + self.robot_half_width * s
            sy = self.robot_half_length * s + self.robot_half_width * c
            if self.mode == 'keep_in':
                if len(self.prisms) != 1:
                    raise ValueError('shape-aware keep-in cost requires one site-boundary prism')
                p = self.prisms[0]
                return ca.mmin(ca.vertcat(
                    m[0] - p.xmin - sx, p.xmax - m[0] - sx,
                    m[1] - p.ymin - sy, p.ymax - m[1] - sy,
                )) - self.body_margin
            dx = ca.fmax(ca.fmax(xmins - sx - m[0], 0.0), m[0] - xmaxs - sx)
            dy = ca.fmax(ca.fmax(ymins - sy - m[1], 0.0), m[1] - ymaxs - sy)
            outside = ca.sqrt(ca.power(dx, 2) + ca.power(dy, 2))
            inside_depth = ca.fmin(
                ca.fmin(m[0] - (xmins - sx), (xmaxs + sx) - m[0]),
                ca.fmin(m[1] - (ymins - sy), (ymaxs + sy) - m[1]))
            signed = ca.mmin(ca.if_else(
                ca.logic_and(dx <= 0.0, dy <= 0.0), -inside_depth, outside))
            return signed - self.body_margin

        def penalty_state_casadi(m):
            deficit = ca.fmax(warning_band - body_clearance(m), 0.0) / warning_band
            overlap = ca.fmax(deficit - 1.0, 0.0)
            return near_weight * (deficit**2 + contact_gain * overlap**2)

        return penalty_state_casadi

    def make_penalty_belief_casadi(self, *, kappa: float = 1.0):
        try:
            import casadi as ca
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError('CasADi is not available for no-go-zone cost') from exc

        if not self.prisms:
            def zero_penalty(_m, _S):
                return 0.0
            return zero_penalty

        # Evaluate the same shape-aware obstacle penalty at five planar sigma
        # points, retaining the predicted mean heading at every point.
        state_penalty = self.make_penalty_state_casadi()
        kappa = max(float(kappa), 1e-6)

        def chol_2x2(M, eps=1e-9):
            M = 0.5 * (M + M.T)
            a = ca.fmax(M[0, 0], eps)
            l11 = ca.sqrt(a)
            l21 = M[1, 0] / l11
            diag22 = ca.fmax(M[1, 1] - l21 * l21, eps)
            l22 = ca.sqrt(diag22)
            return ca.vertcat(
                ca.horzcat(l11, 0.0),
                ca.horzcat(l21, l22),
            )

        def penalty_belief_casadi(m, S):
            mean_xy = ca.reshape(m[:2], 2, 1)
            cov_xy = 0.5 * (S[:2, :2] + S[:2, :2].T)
            spread = math.sqrt(2.0 + kappa) * chol_2x2(cov_xy + 1e-9 * ca.DM.eye(2))
            sigma_points = (
                mean_xy,
                mean_xy + spread[:, 0],
                mean_xy - spread[:, 0],
                mean_xy + spread[:, 1],
                mean_xy - spread[:, 1],
            )
            weights = (
                kappa / (2.0 + kappa),
                1.0 / (2.0 * (2.0 + kappa)),
                1.0 / (2.0 * (2.0 + kappa)),
                1.0 / (2.0 * (2.0 + kappa)),
                1.0 / (2.0 * (2.0 + kappa)),
            )
            total = 0
            for sigma_xy, sigma_weight in zip(sigma_points, weights):
                total += float(sigma_weight) * state_penalty(
                    ca.vertcat(sigma_xy[0], sigma_xy[1], m[2]))
            return total

        return penalty_belief_casadi

    def make_signed_distance_state_casadi(self):
        try:
            import casadi as ca
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError('CasADi is not available for no-go-zone signed distance') from exc

        if not self.prisms:
            def inf_distance(_m):
                return ca.DM.inf()
            return inf_distance

        xmins = ca.DM(self._xmins)
        xmaxs = ca.DM(self._xmaxs)
        ymins = ca.DM(self._ymins)
        ymaxs = ca.DM(self._ymaxs)
        mode = self.mode

        def signed_distance_xy(x, y):
            if mode == 'keep_in' and self.union_boundary_segments:
                q = ca.vertcat(x, y)
                dists = []
                for p1, p2 in self.union_boundary_segments:
                    p1_dm = ca.DM(p1)
                    p2_dm = ca.DM(p2)
                    v = p2_dm - p1_dm
                    w = q - p1_dm
                    v_len_sq = ca.sumsqr(v)
                    t = ca.if_else(v_len_sq < 1e-9, 0.0, ca.fmin(ca.fmax(ca.dot(w, v) / v_len_sq, 0.0), 1.0))
                    closest = p1_dm + t * v
                    dists.append(ca.norm_2(q - closest))
                min_dist = ca.mmin(ca.vertcat(*dists))
                
                is_inside = False
                for p in self.prisms:
                    dx = ca.fmax(ca.fmax(p.xmin - x, 0.0), x - p.xmax)
                    dy = ca.fmax(ca.fmax(p.ymin - y, 0.0), y - p.ymax)
                    is_inside = ca.logic_or(is_inside, ca.logic_and(dx <= 0.0, dy <= 0.0))
                return ca.if_else(is_inside, -min_dist, min_dist)
            else:
                dx = ca.fmax(ca.fmax(xmins - x, 0.0), x - xmaxs)
                dy = ca.fmax(ca.fmax(ymins - y, 0.0), y - ymaxs)
                outside = ca.sqrt(ca.power(dx, 2) + ca.power(dy, 2))

                inside_x = ca.fmin(x - xmins, xmaxs - x)
                inside_y = ca.fmin(y - ymins, ymaxs - y)
                inside_depth = ca.fmin(inside_x, inside_y)
                inside = ca.logic_and(dx <= 0.0, dy <= 0.0)
                signed = ca.if_else(inside, -inside_depth, outside)
                return ca.mmin(signed)

        def signed_distance_state_casadi(m):
            signed_d = signed_distance_xy(m[0], m[1])
            if mode == 'keep_in':
                return -signed_d
            return signed_d

        return signed_distance_state_casadi
