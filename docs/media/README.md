# Media

Animations and stills used in the top-level README and the docs. Every item is
drawn from recorded simulation data or saved model outputs of this thesis; none
is a constructed example. Animation order and pauses are for explanation and
are not the acquisition chronology unless stated.

| File | Shows | Source data |
| --- | --- | --- |
| `camera_network.gif` | The five camera views rendered together in the submitted warehouse world | Gazebo frames recorded in one simulation |
| `simulated_warehouse.png` | Top view of the warehouse with the five camera positions | Thesis setup figure (`figures/make_thesis_setup.py`) |
| `detection_to_floor.gif` | Detector box, bottom-centre point, projected floor position | One recorded camera-B capture frame |
| `reference_pose_collection.gif` | The robot placed at reference poses while four cameras record | Recorded capture session |
| `correction_and_covariance.gif` | Camera-B residuals before and after the learned correction, then the fitted spatial covariance at one query | Saved correction predictions and covariance models |
| `spatial_uncertainty.gif` | The fitted camera-B covariance as a query point moves along a corridor (2σ ellipses) | Saved spatial covariance model; the query motion is a scan, not a drive |
| `route_prediction.gif` | Predicted belief along two candidate routes with camera A removed | Saved planner rollouts (means, covariances, route scores) |
| `closed_loop_navigation.gif` | Camera image, detection and belief versus reference during a drive | Timestamp-matched logs from a recorded lockstep rerun, 1× simulation time |
| `camera_c_dropout_drive.gif` | Camera views during a drive with camera C switched off | Recorded Gazebo frames |
| `camera_c_belief_updates.gif` | Executed route, belief path and fused measurements with camera C off | Recorded run log |
| `system_flowchart.png` | Offline fitting and online navigation loop | Diagram |

Covariance ellipses are 2σ contours (about 86% of the mass in 2D), not 95%
regions, unless the frame says otherwise. The navigation clips illustrate
closed-loop operation; the thesis results are in [`results/`](../../results/).
