"""Split current logs into operational and evaluation-only reliability records."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from reliability.contracts import (
    CameraQuality,
    EvaluationOnlySample,
    OperationalReliabilitySample,
)
from reliability.fusion import MapObservation
from reliability.replay import EvaluationFrame, ReplayFrame
from reliability.single_camera_adapter import (
    SingleCameraAdapterConfig,
    camera_observation_from_diagnostics,
)


@dataclass(frozen=True)
class SplitExport:
    operational_samples: tuple[OperationalReliabilitySample, ...] = field(default_factory=tuple)
    camera_observations: tuple[Any, ...] = field(default_factory=tuple)
    replay_frames: tuple[ReplayFrame, ...] = field(default_factory=tuple)
    evaluation_samples: tuple[EvaluationOnlySample, ...] = field(default_factory=tuple)
    evaluation_frames: tuple[EvaluationFrame, ...] = field(default_factory=tuple)


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def export_split_records_from_rows(
    *,
    perception_rows: Sequence[Mapping[str, Any]],
    experiment_rows: Sequence[Mapping[str, Any]] = (),
    run_id: str = "",
    task_id: str = "",
    seed: int = 0,
    config_hash: str = "",
    artifact_hashes: Mapping[str, str] | None = None,
    camera_config: SingleCameraAdapterConfig | None = None,
    map_measurement_cov_m2: Sequence[Sequence[float]] = ((0.08**2, 0.0), (0.0, 0.08**2)),
) -> SplitExport:
    """Build split records from current logger rows.

    `perception_rows` may contain GT/error columns, but this function only copies
    operational columns into `OperationalReliabilitySample` and camera/replay
    records. GT/outcome/error fields are isolated in evaluation records.
    """

    camera_config = camera_config or SingleCameraAdapterConfig()
    artifact_hashes = dict(artifact_hashes or {})
    operational_samples: list[OperationalReliabilitySample] = []
    camera_observations = []
    replay_frames: list[ReplayFrame] = []
    evaluation_samples: list[EvaluationOnlySample] = []
    evaluation_frames: list[EvaluationFrame] = []

    for idx, row in enumerate(perception_rows):
        stamp = _first_finite(row, ("diag_stamp", "stamp", "log_stamp"), float(idx))
        sample_id = f"{run_id}:{idx:06d}" if run_id else f"sample:{idx:06d}"
        obs = camera_observation_from_diagnostics(
            _perception_row_as_diagnostics(row),
            config=camera_config,
            timestamp_s=stamp,
            measurement_age_s=_first_finite(row, ("pixel_pose_age_s", "frame_age_at_publish_s"), 0.0),
        )
        camera_observations.append(obs)

        operational_samples.append(
            OperationalReliabilitySample(
                sample_id=sample_id,
                timestamp_s=stamp,
                run_id=run_id,
                task_id=task_id,
                seed=seed,
                detector_result=_operational_detector_result(row),
                selected_pixel=obs.pixel_uv,
                projection_valid=_bool(row.get("pixel_pose_available", obs.detection_valid)),
                measurement_age_s=obs.measurement_age_s,
                measurement_stale=not _bool(row.get("pixel_pose_fresh", True)),
                odometry={},
                state_estimate=_state_estimate_summary(row),
                belief=_belief_summary_for_stamp(experiment_rows, stamp),
                camera_relative_range_m=_camera_relative_range(row),
                image_location=_image_location(row),
                recent_detector_history=tuple(),
                config_hash=config_hash,
                artifact_hashes=dict(artifact_hashes),
                metadata={
                    "camera_id": obs.camera_id,
                    "calibration_id": obs.calibration_id,
                    "image_frame_id": obs.image_frame_id,
                    "selected_pixel_source": _str(row.get("yolo_selected_pixel_source", "")),
                },
            )
        )

        map_obs = _map_observation_from_perception_row(row, obs, map_measurement_cov_m2)
        if map_obs is not None:
            replay_frames.append(
                ReplayFrame(
                    timestamp_s=stamp,
                    odometry_xy_m=_odom_xy_for_stamp(experiment_rows, stamp),
                    observations=(map_obs,),
                )
            )

        eval_sample = _evaluation_sample_from_perception_row(
            row,
            sample_id=sample_id,
            timestamp_s=stamp,
            run_id=run_id,
            task_id=task_id,
            seed=seed,
        )
        if eval_sample is not None:
            evaluation_samples.append(eval_sample)
            truth_xy = _truth_xy(row)
            if truth_xy is not None:
                evaluation_frames.append(EvaluationFrame(timestamp_s=stamp, truth_xy_m=truth_xy))

    return SplitExport(
        operational_samples=tuple(operational_samples),
        camera_observations=tuple(camera_observations),
        replay_frames=tuple(replay_frames),
        evaluation_samples=tuple(evaluation_samples),
        evaluation_frames=tuple(evaluation_frames),
    )


def export_multicamera_split_records_from_rows(
    *,
    camera_perception_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    experiment_rows: Sequence[Mapping[str, Any]] = (),
    run_id: str = "",
    task_id: str = "",
    seed: int = 0,
    config_hash: str = "",
    artifact_hashes: Mapping[str, str] | None = None,
    camera_configs: Mapping[str, SingleCameraAdapterConfig] | None = None,
    map_measurement_cov_m2: Sequence[Sequence[float]] = ((0.08**2, 0.0), (0.0, 0.08**2)),
    frame_time_round_digits: int = 3,
) -> SplitExport:
    """Build split records from one perception table per camera.

    This is the extension path for real two-camera logs. It does not infer or use
    ground truth for operational records; any GT/error columns remain isolated in
    evaluation-only samples.
    """

    artifact_hashes = dict(artifact_hashes or {})
    camera_configs = dict(camera_configs or {})
    operational_samples: list[OperationalReliabilitySample] = []
    camera_observations = []
    replay_by_stamp: dict[float, list[MapObservation]] = {}
    evaluation_samples: list[EvaluationOnlySample] = []
    evaluation_frames_by_stamp: dict[float, EvaluationFrame] = {}

    for camera_id, rows in sorted(camera_perception_rows.items()):
        cfg = camera_configs.get(camera_id) or SingleCameraAdapterConfig(
            camera_id=camera_id,
            calibration_id=f"warehouse_multicamera_extension_{camera_id}",
            image_frame_id=camera_id,
        )
        for idx, row in enumerate(rows):
            stamp = _first_finite(row, ("diag_stamp", "stamp", "log_stamp"), float(idx))
            frame_stamp = round(stamp, int(frame_time_round_digits))
            sample_id = f"{run_id}:{camera_id}:{idx:06d}" if run_id else f"{camera_id}:{idx:06d}"
            obs = camera_observation_from_diagnostics(
                _perception_row_as_diagnostics(row),
                config=cfg,
                timestamp_s=stamp,
                measurement_age_s=_first_finite(row, ("pixel_pose_age_s", "frame_age_at_publish_s"), 0.0),
            )
            camera_observations.append(obs)
            operational_samples.append(
                OperationalReliabilitySample(
                    sample_id=sample_id,
                    timestamp_s=stamp,
                    run_id=run_id,
                    task_id=task_id,
                    seed=seed,
                    detector_result=_operational_detector_result(row),
                    selected_pixel=obs.pixel_uv,
                    projection_valid=_bool(row.get("pixel_pose_available", obs.detection_valid)),
                    measurement_age_s=obs.measurement_age_s,
                    measurement_stale=not _bool(row.get("pixel_pose_fresh", True)),
                    odometry={},
                    state_estimate=_state_estimate_summary(row),
                    belief=_belief_summary_for_stamp(experiment_rows, stamp),
                    camera_relative_range_m=_camera_relative_range(row),
                    image_location=_image_location(row),
                    recent_detector_history=tuple(),
                    config_hash=config_hash,
                    artifact_hashes=dict(artifact_hashes),
                    metadata={
                        "camera_id": obs.camera_id,
                        "calibration_id": obs.calibration_id,
                        "image_frame_id": obs.image_frame_id,
                        "selected_pixel_source": _str(row.get("yolo_selected_pixel_source", "")),
                        "frame_time_round_digits": int(frame_time_round_digits),
                    },
                )
            )

            map_obs = _map_observation_from_perception_row(row, obs, map_measurement_cov_m2)
            if map_obs is not None:
                replay_by_stamp.setdefault(frame_stamp, []).append(map_obs)

            eval_sample = _evaluation_sample_from_perception_row(
                row,
                sample_id=sample_id,
                timestamp_s=stamp,
                run_id=run_id,
                task_id=task_id,
                seed=seed,
            )
            if eval_sample is not None:
                evaluation_samples.append(eval_sample)
                truth_xy = _truth_xy(row)
                if truth_xy is not None and frame_stamp not in evaluation_frames_by_stamp:
                    evaluation_frames_by_stamp[frame_stamp] = EvaluationFrame(
                        timestamp_s=frame_stamp,
                        truth_xy_m=truth_xy,
                    )

    replay_frames = [
        ReplayFrame(
            timestamp_s=stamp,
            odometry_xy_m=_odom_xy_for_stamp(experiment_rows, stamp),
            observations=tuple(observations),
        )
        for stamp, observations in sorted(replay_by_stamp.items())
    ]
    return SplitExport(
        operational_samples=tuple(operational_samples),
        camera_observations=tuple(camera_observations),
        replay_frames=tuple(replay_frames),
        evaluation_samples=tuple(evaluation_samples),
        evaluation_frames=tuple(evaluation_frames_by_stamp[key] for key in sorted(evaluation_frames_by_stamp)),
    )


def export_multicamera_run_files(
    *,
    camera_csvs: Mapping[str, str | Path],
    output_dir: str | Path,
    experiment_csv: str | Path | None = None,
    run_id: str = "",
    task_id: str = "",
    seed: int = 0,
    config_hash: str = "",
    artifact_hashes: Mapping[str, str] | None = None,
    frame_time_round_digits: int = 3,
) -> SplitExport:
    camera_rows = {camera_id: read_csv_rows(path) for camera_id, path in camera_csvs.items()}
    experiment_rows = read_csv_rows(experiment_csv) if experiment_csv and Path(experiment_csv).is_file() else []
    export = export_multicamera_split_records_from_rows(
        camera_perception_rows=camera_rows,
        experiment_rows=experiment_rows,
        run_id=run_id,
        task_id=task_id,
        seed=seed,
        config_hash=config_hash,
        artifact_hashes=artifact_hashes,
        frame_time_round_digits=frame_time_round_digits,
    )
    write_split_export(export, output_dir)
    return export


def export_run_directory(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    run_id: str | None = None,
    task_id: str = "",
    seed: int = 0,
    config_hash: str = "",
    artifact_hashes: Mapping[str, str] | None = None,
    camera_config: SingleCameraAdapterConfig | None = None,
) -> SplitExport:
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    perception_path = run_dir / "perception.csv"
    experiment_path = run_dir / "experiment.csv"
    perception_rows = read_csv_rows(perception_path) if perception_path.is_file() else []
    experiment_rows = read_csv_rows(experiment_path) if experiment_path.is_file() else []
    export = export_split_records_from_rows(
        perception_rows=perception_rows,
        experiment_rows=experiment_rows,
        run_id=run_id or run_dir.name,
        task_id=task_id,
        seed=seed,
        config_hash=config_hash,
        artifact_hashes=artifact_hashes,
        camera_config=camera_config,
    )
    write_split_export(export, output_dir)
    return export


def write_split_export(export: SplitExport, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    operational_dir = output_dir / "operational"
    evaluation_dir = output_dir / "evaluation_only"
    operational_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(operational_dir / "operational_reliability.jsonl", export.operational_samples)
    _write_jsonl(operational_dir / "camera_observations.jsonl", export.camera_observations)
    _write_jsonl(operational_dir / "replay_frames.jsonl", export.replay_frames)
    _write_jsonl(evaluation_dir / "evaluation_samples.jsonl", export.evaluation_samples)
    _write_jsonl(evaluation_dir / "evaluation_frames.jsonl", export.evaluation_frames)


def _write_jsonl(path: Path, records: Iterable[Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            if hasattr(record, "to_dict"):
                payload = record.to_dict()
            else:
                payload = _jsonable_dataclass(record)
            handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")


def _jsonable_dataclass(record: Any) -> dict[str, Any]:
    payload = {}
    for key, value in getattr(record, "__dict__", {}).items():
        if isinstance(value, tuple):
            payload[key] = [_jsonable_value(item) for item in value]
        else:
            payload[key] = _jsonable_value(value)
    return payload


def _jsonable_value(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return _jsonable_dataclass(value)
    if isinstance(value, tuple):
        return [_jsonable_value(item) for item in value]
    if isinstance(value, list):
        return [_jsonable_value(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _jsonable_value(v) for k, v in value.items()}
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _perception_row_as_diagnostics(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stamp": row.get("diag_stamp", row.get("stamp", row.get("log_stamp", 0.0))),
        "detected": row.get("detected", 0.0),
        "u_mid": row.get("obs_u", row.get("u_mid", row.get("pixel_pose_u", math.nan))),
        "v_mid": row.get("obs_v", row.get("v_mid", row.get("pixel_pose_v", math.nan))),
        "pixel_pose_u": row.get("pixel_pose_u", math.nan),
        "pixel_pose_v": row.get("pixel_pose_v", math.nan),
        "pixel_pose_age_s": row.get("pixel_pose_age_s", math.nan),
        "yolo_score_raw": row.get("yolo_score_raw", row.get("yolo_raw_best_score", math.nan)),
        "yolo_score_selected": row.get("yolo_score_selected", math.nan),
        "yolo_detected_after_threshold": row.get("yolo_detected_after_threshold", row.get("detected", 0.0)),
        "bbox_xmin": row.get("bbox_xmin", math.nan),
        "bbox_ymin": row.get("bbox_ymin", math.nan),
        "bbox_xmax": row.get("bbox_xmax", math.nan),
        "bbox_ymax": row.get("bbox_ymax", math.nan),
        "frame_age_at_publish_s": row.get("frame_age_at_publish_s", math.nan),
    }


def _operational_detector_result(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "detected": _bool(row.get("detected", False)),
        "yolo_detected_after_threshold": _float_or_none(row.get("yolo_detected_after_threshold")),
        "raw_score": _float_or_none(row.get("yolo_score_raw")),
        "selected_score": _float_or_none(row.get("yolo_score_selected")),
        "bbox_area_px2": _float_or_none(row.get("bbox_area_px")),
        "bbox_xmin": _float_or_none(row.get("bbox_xmin")),
        "bbox_ymin": _float_or_none(row.get("bbox_ymin")),
        "bbox_xmax": _float_or_none(row.get("bbox_xmax")),
        "bbox_ymax": _float_or_none(row.get("bbox_ymax")),
        "mask_area_px": _float_or_none(row.get("mask_area_px")),
        "selected_pixel_source_code": _float_or_none(row.get("selected_pixel_source_code")),
        "inference_latency_ms": _float_or_none(row.get("yolo_inference_ms")),
    }


def _state_estimate_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "available": _bool(row.get("state_available", False)),
        "stamp_s": _float_or_none(row.get("state_age_s")),
        "x_m": _float_or_none(row.get("state_x")),
        "y_m": _float_or_none(row.get("state_y")),
        "yaw_rad": _float_or_none(row.get("state_yaw")),
        "fresh": _bool(row.get("state_fresh", False)),
    }


def _image_location(row: Mapping[str, Any]) -> dict[str, Any]:
    u = _float_or_none(row.get("obs_u", row.get("pixel_pose_u")))
    v = _float_or_none(row.get("obs_v", row.get("pixel_pose_v")))
    return {
        "u_px": u,
        "v_px": v,
        "border_margin_px": _float_or_none(row.get("border_margin_px")),
    }


def _camera_relative_range(row: Mapping[str, Any]) -> float:
    bearing = _float_or_none(row.get("camera_relative_bearing_deg"))
    # Range is not logged directly in current perception.csv; NaN is explicit.
    return math.nan if bearing is None else math.nan


def _map_observation_from_perception_row(
    row: Mapping[str, Any],
    obs,
    map_measurement_cov_m2: Sequence[Sequence[float]],
) -> MapObservation | None:
    x = _float_or_none(row.get("pred_world_x_calibrated", row.get("pred_world_x")))
    y = _float_or_none(row.get("pred_world_y_calibrated", row.get("pred_world_y")))
    if x is None or y is None:
        return None
    quality = CameraQuality(
        camera_id=obs.camera_id,
        p_available=obs.availability_probability,
        conditional_cov_uv=obs.conditional_cov_uv,
        association_confidence=obs.association_probability,
        source_model="single_camera_export",
    )
    return MapObservation(
        camera_id=obs.camera_id,
        timestamp_s=obs.timestamp_s,
        xy_m=(x, y),
        covariance_m2=map_measurement_cov_m2,
        quality=quality,
        source="perception_csv",
    )


def _evaluation_sample_from_perception_row(
    row: Mapping[str, Any],
    *,
    sample_id: str,
    timestamp_s: float,
    run_id: str,
    task_id: str,
    seed: int,
) -> EvaluationOnlySample | None:
    truth_xy = _truth_xy(row)
    loc_err = _float_or_none(row.get("localization_error_calibrated_m", row.get("localization_error_m")))
    if truth_xy is None and loc_err is None:
        return None
    pose = {}
    if truth_xy is not None:
        pose = {
            "x_m": truth_xy[0],
            "y_m": truth_xy[1],
            "yaw_rad": _float_or_none(row.get("true_yaw")),
        }
    return EvaluationOnlySample(
        sample_id=sample_id,
        timestamp_s=timestamp_s,
        run_id=run_id,
        task_id=task_id,
        seed=seed,
        gazebo_ground_truth_pose=pose,
        ground_truth_projected_pixel=None,
        ground_truth_localization_error_m=loc_err,
        clearance_m=math.nan,
        collision=False,
        geometry_breach=False,
        final_task_outcome="",
        metrics={
            "state_error_captime_m": _float_or_none(row.get("state_error_captime_m")),
            "localization_error_captime_m": _float_or_none(row.get("localization_error_captime_m")),
        },
    )


def _truth_xy(row: Mapping[str, Any]) -> tuple[float, float] | None:
    x = _float_or_none(row.get("true_x", row.get("gt_x")))
    y = _float_or_none(row.get("true_y", row.get("gt_y")))
    if x is None or y is None:
        return None
    return (x, y)


def _odom_xy_for_stamp(experiment_rows: Sequence[Mapping[str, Any]], stamp_s: float) -> tuple[float, float]:
    if not experiment_rows:
        return (0.0, 0.0)
    best = min(
        experiment_rows,
        key=lambda row: abs(_first_finite(row, ("stamp", "log_stamp"), stamp_s) - stamp_s),
    )
    x = _float_or_none(best.get("odom_noisy_x", best.get("odom_x", best.get("odom_map_x"))))
    y = _float_or_none(best.get("odom_noisy_y", best.get("odom_y", best.get("odom_map_y"))))
    return (x or 0.0, y or 0.0)


def _belief_summary_for_stamp(
    experiment_rows: Sequence[Mapping[str, Any]], stamp_s: float
) -> dict[str, Any]:
    """Return nearest operational noisy-odometry belief without any truth fields."""

    if not experiment_rows:
        return {}
    best = min(
        experiment_rows,
        key=lambda row: abs(_first_finite(row, ("stamp", "log_stamp"), stamp_s) - stamp_s),
    )
    x = _float_or_none(best.get("odom_noisy_x", best.get("odom_x", best.get("odom_map_x"))))
    y = _float_or_none(best.get("odom_noisy_y", best.get("odom_y", best.get("odom_map_y"))))
    if x is None or y is None:
        return {}
    payload: dict[str, Any] = {
        "source": "operational_noisy_odometry",
        "stamp_s": _first_finite(best, ("stamp", "log_stamp"), stamp_s),
        "x_m": x,
        "y_m": y,
        "alignment_age_s": abs(_first_finite(best, ("stamp", "log_stamp"), stamp_s) - stamp_s),
    }
    xx = _float_or_none(best.get("odom_noisy_cov_xx"))
    xy = _float_or_none(best.get("odom_noisy_cov_xy"))
    yy = _float_or_none(best.get("odom_noisy_cov_yy"))
    if xx is not None and xy is not None and yy is not None and xx > 0.0 and yy > 0.0 and xx * yy > xy * xy:
        payload["covariance_xy_m2"] = [[xx, xy], [xy, yy]]
    return payload


def _first_finite(row: Mapping[str, Any], keys: Sequence[str], default: float) -> float:
    for key in keys:
        value = _float_or_none(row.get(key))
        if value is not None:
            return value
    return float(default)


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    numeric = _float_or_none(value)
    if numeric is not None:
        return numeric >= 0.5
    return bool(value)


def _str(value: Any) -> str:
    return "" if value is None else str(value)
