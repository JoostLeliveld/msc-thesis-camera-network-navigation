# `unav_common`

This package contains shared geometry and manifest utilities used by multiple active packages.

## Main Files

| File | Role |
| --- | --- |
| [`unav_common/camera_model.py`](unav_common/camera_model.py) | camera projection and inverse projection helpers |
| [`unav_common/occlusion_geometry.py`](unav_common/occlusion_geometry.py) | world-geometry parsing and occlusion support |
| [`unav_common/manifest.py`](unav_common/manifest.py) | manifest helpers used by experiment logging |
| [`unav_common/geometry.py`](unav_common/geometry.py) | general geometry helpers |

## Who Uses This Package

- `perception`
- `state`
- `planning`
- `experiments`

## Caveat

This is support code. It is important for understanding camera geometry and world geometry, but it is not where the thesis comparison itself is defined.
