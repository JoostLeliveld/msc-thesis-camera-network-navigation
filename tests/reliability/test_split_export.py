from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "reliability"))

from reliability import (  # noqa: E402
    export_multicamera_split_records_from_rows,
    export_split_records_from_rows,
    validate_camera_overlap,
    write_split_export,
)


def _perception_rows() -> list[dict[str, str]]:
    return [
        {
            "diag_stamp": "1.0",
            "log_stamp": "1.1",
            "detected": "1",
            "true_x": "0.9",
            "true_y": "0.1",
            "true_yaw": "0.0",
            "state_available": "1",
            "state_x": "1.0",
            "state_y": "0.0",
            "state_yaw": "0.0",
            "state_age_s": "0.05",
            "state_fresh": "1",
            "obs_u": "392.0",
            "obs_v": "456.0",
            "pixel_pose_available": "1",
            "pixel_pose_u": "392.0",
            "pixel_pose_v": "456.0",
            "pixel_pose_age_s": "0.04",
            "pixel_pose_fresh": "1",
            "pred_world_x_calibrated": "0.95",
            "pred_world_y_calibrated": "0.05",
            "localization_error_calibrated_m": "0.071",
            "localization_error_captime_m": "0.05",
            "state_error_captime_m": "0.10",
            "yolo_score_raw": "0.80",
            "yolo_score_selected": "0.82",
            "yolo_detected_after_threshold": "1",
            "bbox_area_px": "3000",
            "bbox_xmin": "360",
            "bbox_ymin": "410",
            "bbox_xmax": "424",
            "bbox_ymax": "456",
            "selected_pixel_source_code": "1",
            "yolo_selected_pixel_source": "bbox_bottom",
            "yolo_inference_ms": "18",
        }
    ]


def test_export_split_records_keeps_operational_and_evaluation_fields_apart() -> None:
    export = export_split_records_from_rows(
        perception_rows=_perception_rows(),
        experiment_rows=[{"stamp": "1.0", "odom_noisy_x": "1.0", "odom_noisy_y": "0.0"}],
        run_id="run_001",
        task_id="task",
        seed=7,
        config_hash="sha256:cfg",
        artifact_hashes={"yolo": "sha256:yolo"},
    )

    assert len(export.operational_samples) == 1
    assert len(export.evaluation_samples) == 1
    assert len(export.camera_observations) == 1
    assert len(export.replay_frames) == 1
    operational = export.operational_samples[0].to_dict()
    evaluation = export.evaluation_samples[0].to_dict()

    assert operational["selected_pixel"] == [392.0, 456.0]
    assert "true_x" not in json.dumps(operational)
    assert "localization_error" not in json.dumps(operational)
    assert evaluation["gazebo_ground_truth_pose"]["x_m"] == 0.9
    assert evaluation["ground_truth_localization_error_m"] == 0.071


def test_write_split_export_creates_expected_jsonl_files(tmp_path: Path) -> None:
    export = export_split_records_from_rows(perception_rows=_perception_rows(), run_id="run_001")
    write_split_export(export, tmp_path)

    operational_path = tmp_path / "operational" / "camera_observations.jsonl"
    evaluation_path = tmp_path / "evaluation_only" / "evaluation_samples.jsonl"

    assert operational_path.is_file()
    assert evaluation_path.is_file()
    assert json.loads(operational_path.read_text(encoding="utf-8").splitlines()[0])["camera_id"] == "camera_A"
    assert json.loads(evaluation_path.read_text(encoding="utf-8").splitlines()[0])["run_id"] == "run_001"


def test_multicamera_export_builds_shared_replay_frames() -> None:
    camera_a = _perception_rows()
    camera_b = [dict(camera_a[0])]
    camera_b[0]["diag_stamp"] = "1.0004"
    camera_b[0]["pred_world_x_calibrated"] = "1.02"
    camera_b[0]["pred_world_y_calibrated"] = "0.04"
    camera_b[0]["obs_u"] = "612.0"
    camera_b[0]["obs_v"] = "388.0"

    export = export_multicamera_split_records_from_rows(
        camera_perception_rows={"camera_A": camera_a, "camera_B": camera_b},
        experiment_rows=[{
            "stamp": "1.0", "odom_noisy_x": "1.0", "odom_noisy_y": "0.0",
            "odom_noisy_cov_xx": "0.04", "odom_noisy_cov_xy": "0.01", "odom_noisy_cov_yy": "0.09",
        }],
        run_id="run_multi",
        frame_time_round_digits=3,
    )

    assert len(export.operational_samples) == 2
    assert len(export.camera_observations) == 2
    assert len(export.evaluation_frames) == 1
    assert len(export.replay_frames) == 1
    assert export.operational_samples[0].belief == {
        "source": "operational_noisy_odometry",
        "stamp_s": 1.0,
        "x_m": 1.0,
        "y_m": 0.0,
        "alignment_age_s": 0.0,
        "covariance_xy_m2": [[0.04, 0.01], [0.01, 0.09]],
    }
    frame = export.replay_frames[0]
    assert {obs.camera_id for obs in frame.observations} == {"camera_A", "camera_B"}

    summary = validate_camera_overlap(
        export.replay_frames,
        max_time_delta_s=0.01,
        max_allowed_disagreement_m=0.10,
    )
    assert summary.pass_gate is True
    assert summary.pair_count == 1
