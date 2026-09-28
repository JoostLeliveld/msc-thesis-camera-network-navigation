# Perception

This package runs the frozen external-camera YOLO detector. Its paper-facing output is a
bounding box and detector metadata for each identified camera frame. The runtime selects
the box bottom centre; geometric projection and commissioned correction happen downstream.

The detector does not estimate heading and does not construct a robot hull. Detector
training and dataset provenance live under `scripts/perception/` and the locked Stage-04/05
manifests under `pipeline/`.
