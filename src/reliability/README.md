# reliability

Turns detections into corrected ground-plane measurements with a covariance, and fuses
simultaneous cameras.

- `reliability/nodes/camera_manager_node.py`: runtime node. For each camera frame it
  applies the observation gate, projects the box bottom centre to the ground plane,
  applies the learned correction, queries the covariance model and fuses the admitted
  cameras in information form.
- `reliability/observation_gates.py`: the fixed confidence and projection gate.
- `reliability/visibility_residual_net.py`: the correction network, shared by fitting
  (`pipeline/fit_correction.py`) and runtime.
- `reliability/measurement_fusion.py`: batching and fusion of simultaneous cameras.

The correction and the global, per-camera and spatial covariance models are fitted by
`pipeline/refit.sh`.
