"""The floor on belief uncertainty that repeated looks must not be able to shrink.

Why this exists
---------------
A Kalman filter treats N sightings of one place as N independent votes and contracts
like 1/N. Part of this system's error is not a vote: it repeats. Averaging ten sightings
of the same pose from the same camera removes the scatter and leaves the repeatable part
untouched, so a filter that models only scatter becomes confident and wrong -- which is
the failure this whole study is organised around.

`DESIGN_LOCK.md` D5 calls this layer 3 and has done since the method was written. It was
never implemented: the only floor in the runtime was `_psd_floor`, a numerical epsilon
that keeps a matrix invertible and has nothing to do with bias.

Shape, not just size
--------------------
Two measurements decide what the floor looks like, and both say a scalar is wrong.

D21: the residual is almost entirely ALONG the camera's line of sight -- across-ray means
stay under a centimetre in every camera while along-ray reaches six -- and it grows with
range at roughly 0.2 degrees equivalent in all five cameras. So the floor is anisotropic
and oriented by the ray, exactly like `R` itself.

D22: the smallest proportional bound above every measured (camera, range) cell is
**1.6 mm per metre**, which is 0.96 cm at 6 m and 3.20 cm at 20 m. The flat 2.5 cm that
the design lock carried is 2.6x too large at 6 m and below the envelope at 20 m: it
over-bounds where the cameras are good and under-bounds where they are weakest.

Where it applies
----------------
On the BELIEF, after the update -- not on `R`. Adding to `R` re-brands a persistent error
as fresh per-frame noise, so the filter still shrinks it like 1/N; it only starts from a
larger number. Flooring the posterior is what makes repetition unable to shrink it.

This is not a second copy of `reliability.covariance_mapping`, which maps a trust scalar
to a measurement covariance. That answers "how noisy is this sighting"; this answers "how
certain is the filter allowed to become", and they compose.
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

Matrix2x2 = tuple[tuple[float, float], tuple[float, float]]

#: metres of along-ray bias per metre of range. D22's bounding envelope over every
#: measured (camera, range) cell of the frozen detector reading. Not a fit: half the
#: cells sit above any fit by construction, and a floor is a bound.
DEFAULT_ALONG_RAY_SLOPE = 0.0016

#: across-ray bias per metre of range. D21 measures 0.010 to 0.052 degrees against
#: 0.175 to 0.285 along-ray, so the across component is real but roughly a fifth.
DEFAULT_ACROSS_RAY_SLOPE = 0.00035

#: the flat floor this replaces, kept only so a comparison arm can ask for it by name
RETIRED_CONSTANT_FLOOR_M = 0.025


class BiasFloorError(ValueError):
    """Raised when a floor cannot be formed from the inputs given."""


def ray_bearing_rad(camera_xy: Sequence[float], target_xy: Sequence[float]) -> float:
    """Bearing from camera to target in the world frame."""
    dx = float(target_xy[0]) - float(camera_xy[0])
    dy = float(target_xy[1]) - float(camera_xy[1])
    if not (math.isfinite(dx) and math.isfinite(dy)):
        raise BiasFloorError("camera and target positions must be finite")
    if dx == 0.0 and dy == 0.0:
        raise BiasFloorError("target coincides with the camera; no ray exists")
    return math.atan2(dy, dx)


def bias_floor_matrix(
    range_m: float,
    bearing_rad: float,
    *,
    along_slope: float = DEFAULT_ALONG_RAY_SLOPE,
    across_slope: float = DEFAULT_ACROSS_RAY_SLOPE,
) -> Matrix2x2:
    """The 2x2 world-frame covariance floor for a sighting at this range and bearing.

    Built in the ray frame, where the measured structure lives, then rotated into the
    world frame. Returns a covariance (metres squared), not a standard deviation.
    """
    rho = float(range_m)
    if not math.isfinite(rho) or rho < 0.0:
        raise BiasFloorError(f"range must be finite and non-negative, got {range_m!r}")
    if not math.isfinite(float(bearing_rad)):
        raise BiasFloorError("bearing must be finite")
    for name, slope in (("along_slope", along_slope), ("across_slope", across_slope)):
        if not math.isfinite(float(slope)) or float(slope) < 0.0:
            raise BiasFloorError(f"{name} must be finite and non-negative")

    sigma_along = float(along_slope) * rho
    sigma_across = float(across_slope) * rho
    a2, c2 = sigma_along ** 2, sigma_across ** 2
    cos_b, sin_b = math.cos(float(bearing_rad)), math.sin(float(bearing_rad))
    # R diag(a2, c2) R^T with R the ray->world rotation
    xx = a2 * cos_b * cos_b + c2 * sin_b * sin_b
    yy = a2 * sin_b * sin_b + c2 * cos_b * cos_b
    xy = (a2 - c2) * cos_b * sin_b
    return ((xx, xy), (xy, yy))


def apply_belief_floor(belief_cov: Matrix2x2, floor: Matrix2x2) -> Matrix2x2:
    """Raise a belief covariance so it is at least the floor in every direction.

    Implemented as a symmetric-eigenvalue floor of `floor^-1/2 P floor^-1/2`, which is
    the statement "the belief may not be sharper than the floor along ANY direction".
    A cheaper elementwise maximum does not mean that: two matrices can each have larger
    diagonals while one is still sharper along a diagonal direction.
    """
    p = _validated(belief_cov, "belief_cov")
    f = _validated(floor, "floor")
    # whiten by the floor: eigenvalues below 1 are directions the belief is too sharp in
    w, vectors = _sym_eig(f)
    if min(w) <= 0.0:
        raise BiasFloorError("floor must be positive definite")
    inv_sqrt = _compose(vectors, tuple(1.0 / math.sqrt(value) for value in w))
    sqrt = _compose(vectors, tuple(math.sqrt(value) for value in w))
    whitened = _mul(_mul(inv_sqrt, p), inv_sqrt)
    lam, q = _sym_eig(whitened)
    raised = _compose(q, tuple(max(value, 1.0) for value in lam))
    return _mul(_mul(sqrt, raised), sqrt)


def _validated(matrix: Matrix2x2, name: str) -> Matrix2x2:
    try:
        (a, b), (c, d) = matrix
        a, b, c, d = float(a), float(b), float(c), float(d)
    except Exception as exc:  # noqa: BLE001
        raise BiasFloorError(f"{name} must be a 2x2 matrix") from exc
    if not all(math.isfinite(v) for v in (a, b, c, d)):
        raise BiasFloorError(f"{name} entries must be finite")
    if abs(b - c) > 1e-9 * max(1.0, abs(b), abs(c)):
        raise BiasFloorError(f"{name} must be symmetric")
    return ((a, b), (c, d))


def _sym_eig(matrix: Matrix2x2):
    (a, b), (_b, d) = matrix
    tr, det = a + d, a * d - b * b
    disc = max(tr * tr / 4.0 - det, 0.0)
    root = math.sqrt(disc)
    l1, l2 = tr / 2.0 + root, tr / 2.0 - root
    if abs(b) > 1e-18:
        v1 = (l1 - d, b)
        n1 = math.hypot(*v1)
        v1 = (v1[0] / n1, v1[1] / n1)
    else:
        v1 = (1.0, 0.0) if a >= d else (0.0, 1.0)
    v2 = (-v1[1], v1[0])
    return (l1, l2), (v1, v2)


def _compose(vectors, values) -> Matrix2x2:
    (v1, v2), (l1, l2) = vectors, values
    xx = l1 * v1[0] * v1[0] + l2 * v2[0] * v2[0]
    yy = l1 * v1[1] * v1[1] + l2 * v2[1] * v2[1]
    xy = l1 * v1[0] * v1[1] + l2 * v2[0] * v2[1]
    return ((xx, xy), (xy, yy))


def _mul(left: Matrix2x2, right: Matrix2x2) -> Matrix2x2:
    (a, b), (c, d) = left
    (e, f), (g, h) = right
    return ((a * e + b * g, a * f + b * h), (c * e + d * g, c * f + d * h))


def combine_floors(floors: Sequence[Matrix2x2]) -> Matrix2x2:
    """A floor at least as wide as EVERY contributing sighting's floor.

    The first version of this combined floors by inverse sum, the way independent
    covariances combine, on the reasoning that each camera's bias points along its own ray
    so different bearings partly cancel. That was measured and it is wrong, badly: a single
    camera's floor at 15 m is 2.40 cm along its ray, and inverse-summing two of them gives
    **0.61 cm** -- a fourfold *reduction*. Three cameras at 20 m take a 3.20 cm floor down
    to 0.60 cm.

    The algebra is right and the model is wrong. These floors are very anisotropic (2.40 cm
    along the ray, 0.53 across), so inverse-summing lets one camera's across-ray
    *tightness* constrain the direction another camera is loose in. But camera A being
    precise across its own ray is not information about camera B's along-ray bias. Treating
    a bias bound as independent information lets two cameras average away an error that is
    not independent in the way the algebra assumes -- which is exactly the unearned
    confidence this floor exists to forbid.

    So the rule is conservative: the result is at least as wide as every input, in every
    direction. Built by flooring one input with the next in turn, which is monotone --
    `apply_belief_floor` only ever raises eigenvalues, so an earlier floor cannot be undone
    by a later one.

    The cost of being conservative is over-coverage when viewpoints really are
    complementary. That is the right direction to err for a bound whose entire purpose is
    to stop a filter claiming precision it has not earned, and it is checked by test rather
    than argued.
    """
    if not floors:
        raise BiasFloorError("at least one floor is required")
    combined = _validated(floors[0], "floor")
    w, _ = _sym_eig(combined)
    if min(w) <= 0.0:
        raise BiasFloorError("every floor must be positive definite")
    for floor in floors[1:]:
        f = _validated(floor, "floor")
        w, _ = _sym_eig(f)
        if min(w) <= 0.0:
            raise BiasFloorError("every floor must be positive definite")
        combined = apply_belief_floor(combined, f)
    return combined


def combine_camera_floors(
    floors: Mapping[str, Matrix2x2], camera_ids: Iterable[str],
) -> Matrix2x2 | None:
    """Combine contributing camera bounds once in stable camera-ID order.

    Pairwise covariance flooring is not associative for three noncommuting
    matrices. The ordering is part of this conservative bound, not callback
    arrival order; it does not turn the bound into independent information.
    """
    chosen = [floors[camera_id] for camera_id in sorted(set(camera_ids)) if camera_id in floors]
    return combine_floors(chosen) if chosen else None
