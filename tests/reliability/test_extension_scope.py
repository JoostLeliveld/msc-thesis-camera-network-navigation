from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_detector_topic_parameters_keep_single_camera_defaults() -> None:
    detector = (ROOT / "src" / "perception" / "perception" / "nodes" / "yolo_robot_detector_node.py").read_text(
        encoding="utf-8"
    )

    assert "declare_parameter('image_topic', '/external_camera/image_raw')" in detector
    assert "declare_parameter('pixel_pose_topic', '/perception/pixel_pose')" in detector
    assert "declare_parameter('diagnostics_topic', DETECTION_DIAGNOSTICS_TOPIC)" in detector
    assert "declare_parameter('camera_observation_topic', '')" in detector
    assert "self.create_subscription(Image, self.image_topic" in detector
    assert "self.create_publisher(PoseStamped, self.pixel_pose_topic" in detector
