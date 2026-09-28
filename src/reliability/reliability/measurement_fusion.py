"""Prior-free camera-batch admission and aggregation, shared by runtime and replay.

The disagreement threshold and aggregation rule are explicit experiment inputs.
This module does not choose a state prior, change camera R, or infer motion.
"""
from dataclasses import dataclass, field
import math
from statistics import median

from reliability.contracts import ContractValidationError
from reliability.fusion import (
    camera_measurement_batch,
    distance_angle_weighted_fusion_2d,
    independent_measurement_fusion_2d,
    joint_network_estimate_2d,
    select_smallest_covariance,
)

FUSION_RULE_BEST_SINGLE = "best_single"
FUSION_RULE_DISTANCE_ANGLE = "distance_angle"
FUSION_RULE_INDEPENDENT = "independent"
FUSION_RULE_JOINT_NETWORK = "joint_network"
SUPPORTED_FUSION_RULES = (
    FUSION_RULE_BEST_SINGLE, FUSION_RULE_DISTANCE_ANGLE,
    FUSION_RULE_INDEPENDENT, FUSION_RULE_JOINT_NETWORK,
)


@dataclass(frozen=True)
class MeasurementFusionResult:
    """Used camera IDs and metric gate residuals; no measurement if all are refused.

    These residuals are distances from the batch median in metres. They are
    not NIS, which requires an innovation covariance and belongs to a filter.
    """
    mean_xy: tuple[float, float] | None
    covariance_m2: tuple[tuple[float, float], tuple[float, float]] | None
    accepted_camera_ids: tuple[str, ...]
    rejected_camera_ids: tuple[str, ...]
    residuals_m_by_camera: dict[str, float] = field(default_factory=dict)


def combine_measurements_2d(accepted, *, rule: str, camera_positions_m=None):
    """Aggregate an already admitted batch using one declared rule."""
    _validate_rule(rule, camera_positions_m)
    accepted = camera_measurement_batch(accepted)
    if rule == FUSION_RULE_BEST_SINGLE:
        chosen = select_smallest_covariance(accepted)
        return chosen.xy_m, chosen.covariance_m2, (chosen.camera_id,)
    if rule == FUSION_RULE_DISTANCE_ANGLE:
        mean, covariance = distance_angle_weighted_fusion_2d(accepted, camera_positions_m)
    elif rule == FUSION_RULE_JOINT_NETWORK:
        mean, covariance = joint_network_estimate_2d(accepted)
    else:  # validated independent rule
        mean, covariance = independent_measurement_fusion_2d(accepted)
    return mean, covariance, tuple(observation.camera_id for observation in accepted)


def _validate_rule(rule, camera_positions_m):
    if rule not in SUPPORTED_FUSION_RULES:
        raise ContractValidationError(f"unsupported fusion_rule {rule!r}; expected {SUPPORTED_FUSION_RULES}")
    if rule == FUSION_RULE_DISTANCE_ANGLE and camera_positions_m is None:
        raise ContractValidationError("the distance_angle rule needs camera positions")


def gated_measurement_fusion_2d(
    observations, *, disagreement_gate_m: float, rule: str,
    camera_positions_m=None, belief_floors=None,
) -> MeasurementFusionResult:
    """Gate around the component median, then combine survivors by ``rule``.

    Identity/configuration errors are refused before statistical decisions. A
    batch whose cameras all fail the distance gate returns no measurement.
    Optional floors apply once to the result, using only contributing cameras.
    """
    _validate_rule(rule, camera_positions_m)
    observations = camera_measurement_batch(observations)
    gate = float(disagreement_gate_m)
    # math.inf disables the median gate: every admitted camera is fused.
    if math.isnan(gate) or gate <= 0.0:
        raise ContractValidationError("disagreement_gate_m must be positive (math.inf: no gate)")
    centre = tuple(median(obs.xy_m[axis] for obs in observations) for axis in (0, 1))
    residuals = {obs.camera_id: math.hypot(obs.xy_m[0]-centre[0], obs.xy_m[1]-centre[1])
                 for obs in observations}
    accepted = [obs for obs in observations if residuals[obs.camera_id] <= gate]
    rejected = tuple(obs.camera_id for obs in observations if residuals[obs.camera_id] > gate)
    if not accepted:
        return MeasurementFusionResult(None, None, (), rejected, residuals)
    mean, covariance, used = combine_measurements_2d(
        accepted, rule=rule, camera_positions_m=camera_positions_m,
    )
    if belief_floors:
        from reliability.bias_floor import apply_belief_floor, combine_camera_floors

        floor = combine_camera_floors(belief_floors, used)
        if floor is not None:
            covariance = apply_belief_floor(covariance, floor)
    return MeasurementFusionResult(mean, covariance, used, rejected, residuals)
