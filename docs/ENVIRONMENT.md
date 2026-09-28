# Environment and verification scope

The supported platform is Ubuntu 22.04, Python 3.10, ROS 2 Humble and Gazebo
Fortress. The following package versions were installed on the submission-review
machine on 28 September 2026. This is an environment inventory, not a lockfile
or a claim that it recreates every historical capture environment.

| Package | Review environment |
| --- | --- |
| numpy | 1.26.4 |
| scipy | 1.15.3 |
| matplotlib | 3.10.8 |
| casadi | 3.7.2 |
| scikit-learn | 1.7.2 |
| opencv-python | 4.13.0.92 |
| PyYAML | 5.4.1 |
| torch | 2.5.1+cu118 |
| torchvision | 0.20.1+cu118 |
| ultralytics | 8.4.36 |
| pandas | 2.3.3 |
| joblib | 1.5.3 |
| Pillow | 12.1.0 |
| pytest | 6.2.5 |

The review machine has mixed apt/pip packages. In particular, its OpenCV 4.13
package metadata requires NumPy 2 even though NumPy 1.26 is installed. The
submission requirements therefore bound OpenCV below 4.12 and NumPy below 2,
and request PyYAML 6, to avoid carrying these inconsistencies into a new
Humble environment. The complete Linux/Python 3.10 resolution is pinned in
`requirements-lock.txt`; pip resolved it successfully without installing it
on the review machine. Run `python3 -m pip check` after installation.

The code-only test suite and recorded-data analyses are checked separately.
No claim is made that a fresh 90-run simulation campaign was executed for the
submission cleanup. Rendering, detector kernels and optimisation libraries can
vary across hardware or package builds; archived logs remain the reference for
the reported experiment. Capture `python3 -m pip freeze`, the driver information
and `git rev-parse HEAD` when reporting a new run.
