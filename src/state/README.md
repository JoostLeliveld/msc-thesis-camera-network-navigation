# state

Pixel-to-ground-plane conversion for single-camera use.

- `state/core/pixel_to_bev.py`: projects an image point to the ground plane with the
  camera homography.
- `state/nodes/pixel_to_bev_state_node.py`: node for the single-camera path.

The thesis campaign uses the multi-camera path in `reliability/`; this package supplies
the shared projection code.
