"""Oriented chassis geometry and conservative continuous pose-segment checks.

Positive clearance is exact Euclidean separation. Negative clearance denotes
overlap (its magnitude is not a minimum translation distance). Sweeps certify
linear XY / unwrapped-yaw interpolation and, when supplied, constant-command
unicycle arcs; not unmodelled actuation or braking.
"""
import math

import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union


def constant_twist_pose(start, control, elapsed):
    """Exact planar pose under a held linear/angular velocity command."""
    start = np.asarray(start, dtype=float)
    v, w = control
    angle = w*elapsed
    distance = v*elapsed*float(np.sinc(angle/(2*math.pi)))
    direction = start[2]+angle/2
    return start+np.array([distance*math.cos(direction), distance*math.sin(direction), angle])


class RectangularFootprint:
    def __init__(self, prisms, length=.8, width=.55, *, keep_in=False):
        if not all(math.isfinite(v) and v > 0 for v in (length, width)):
            raise ValueError('robot length and width must be finite and positive')
        self.length, self.width = float(length), float(width)
        self.radius = math.hypot(length/2, width/2)
        self.keep_in = keep_in
        rectangles = []
        for p in prisms:
            bounds = (p.xmin, p.ymin, p.xmax, p.ymax)
            if not all(math.isfinite(v) for v in bounds) or p.xmin >= p.xmax or p.ymin >= p.ymax:
                raise ValueError('invalid footprint scene rectangle')
            rectangles.append(box(*bounds))
        self.scene = unary_union(rectangles)

    def polygon(self, pose):
        x, y, yaw = np.asarray(pose, dtype=float)[:3]
        if not np.isfinite([x, y, yaw]).all():
            raise ValueError('nonfinite footprint pose')
        c, s = math.cos(yaw), math.sin(yaw)
        corners = [(x+c*dx-s*dy, y+s*dx+c*dy)
                   for dx, dy in ((-self.length/2, -self.width/2),
                                  (self.length/2, -self.width/2),
                                  (self.length/2, self.width/2),
                                  (-self.length/2, self.width/2))]
        return Polygon(corners)

    def clearance(self, pose):
        body = self.polygon(pose)
        if self.scene.is_empty:
            return -math.inf if self.keep_in else math.inf
        if self.keep_in:
            if self.scene.covers(body):
                return float(body.distance(self.scene.boundary))
            return -max(math.sqrt(body.difference(self.scene).area), 1e-12)
        if body.intersects(self.scene):
            return -max(math.sqrt(body.intersection(self.scene).area), 1e-12)
        return float(body.distance(self.scene))

    def sweep_clearance(self, start, end, *, yaw_delta=None, max_depth=16, control=None,
                        dt=None, required_clearance=0.0):
        """Return a certified lower clearance bound, or negative on refusal.

        Every body point moves by at most centre travel + radius * yaw travel.
        A midpoint clearance larger than that interval's half-travel certifies
        the entire interval. Otherwise subdivide; unresolved intervals fail
        closed. This catches thin obstacles and rotation with clear endpoints.
        """
        start, end = np.asarray(start, dtype=float)[:3], np.asarray(end, dtype=float)[:3]
        required_clearance = float(required_clearance)
        if not math.isfinite(required_clearance) or required_clearance < 0.0:
            raise ValueError('required_clearance must be finite and non-negative')
        if start.shape != (3,) or end.shape != (3,) or not np.isfinite([start, end]).all():
            return -math.inf
        delta = end-start
        delta[2] = ((delta[2]+math.pi) % (2*math.pi)-math.pi) if yaw_delta is None else yaw_delta
        if not np.isfinite(delta).all():
            return -math.inf
        nominal_bound = math.inf
        path = lambda fraction: start+fraction*delta
        travel = float(np.linalg.norm(delta[:2])) + self.radius*abs(delta[2])
        if control is not None:
            command = np.asarray(control, dtype=float)
            if command.shape != (2,) or not np.isfinite(command).all() or dt is None or not math.isfinite(dt) or dt <= 0:
                return -math.inf
            v, w = command
            # Certify both the planner's Euler pose segment and the exact
            # constant-twist arc; do not silently change estimator dynamics.
            nominal_bound = self.sweep_clearance(
                start, end, yaw_delta=w*dt, max_depth=max_depth,
                required_clearance=required_clearance)
            if nominal_bound < required_clearance:
                return nominal_bound
            def path(fraction):
                return constant_twist_pose(start, command, fraction*dt)
            travel = abs(v)*dt + self.radius*abs(w)*dt
        endpoint_clearance = min(self.clearance(start), self.clearance(path(1.)))
        if endpoint_clearance < required_clearance or travel == 0:
            return endpoint_clearance
        stack = [(0., 1., 0)]
        certified = endpoint_clearance
        while stack:
            lo, hi, depth = stack.pop()
            mid = (lo+hi)/2
            clearance = self.clearance(path(mid))
            if clearance < required_clearance:
                return clearance
            bound = travel*(hi-lo)/2
            if clearance - bound >= required_clearance:
                certified = min(certified, clearance-bound)
            elif depth >= max_depth:
                return -math.inf
            else:
                stack.extend(((lo, mid, depth+1), (mid, hi, depth+1)))
        return min(certified, nominal_bound)
