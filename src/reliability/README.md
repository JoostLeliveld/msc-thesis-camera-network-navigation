# Reliability and camera-network sensing

This package contains the paper-facing camera-network measurement path.

The active contract is:

1. a frozen YOLO box is admitted by a fixed, belief-independent sensor gate;
2. the raw observation is the box bottom centre projected to the ground plane;
3. the selected correction is either the raw-box baseline, a box-feature MLP residual, or
   the same MLP with a gated image residual;
4. each correction uses its own covariance model fitted from whole-drive out-of-fold
   residuals;
5. planner precision is the direct inverse of the matched runtime covariance, queried for
   the same camera and position and rotated into the common world frame;
6. a single robot filter consumes each camera frame once.

No second planning field, availability model, opportunity outcome, detector miss, admission
refusal, or NIS result modifies that precision. Weak spatial support affects M2 through its
broad covariance prior, so unsupported regions contribute little precision. Visual-hull
observations, hull-derived correction inputs, belief-dependent admission, and process-noise
selection are outside the thesis method. See `docs/METHOD.md` for the
normative definitions.

Current runtime entry points are `reliability/nodes/camera_manager_node.py`,
`reliability/commissioned_availability.py`, and `reliability/commissioned_visibility.py`.
