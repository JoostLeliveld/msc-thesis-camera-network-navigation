# perception

Runs the frozen YOLO11n robot detector on the camera images.

- `perception/nodes/batched_four_camera_yolo_node.py`: the detector node used in the
  campaign. It processes the frames of all cameras in one batch and publishes the
  bounding box and confidence of each detection.
- `perception/core/`: frame batching, detector outcomes and diagnostics.

The detector is trained by `pipeline/detector/`. The trained checkpoint is not in Git
(see the data section of the root README).
