# Setup image

`gazebo_plan_view.png` is the captured Gazebo image used as the background of
the thesis setup illustration. It was recovered from the thesis presentation's
shared visual assets; it is a Gazebo rendering. `make_thesis_setup.py` adds camera labels. This bundled raster is already
cropped and resampled (1078 x 902 pixels); it represents the original
1600 x 1200 capture cropped to x=185:1415, y=85:1115. The generator preserves
that coordinate mapping rather than cropping it twice. See `capture_plan_view.py` to capture a replacement.

Rendered scenery retains the third-party asset attributions documented in
`src/sim/fetch_external_models.sh` and the source-model notices. In particular,
the crates use the OpenRobotics Large Crate asset under CC-BY 4.0.
