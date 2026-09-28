#!/bin/bash
# The reference fits, in order: detector inference (frozen YOLO11n), sensor gate, correction
# (CPU, deterministic), corrected residuals, R0/R1/R2 (K=16, l=0.4 m fixed), D_dev evaluation
# incl. R_proj, runtime package and planning precision. Each stage refuses to overwrite its
# output, so a rerun skips finished stages and resumes where it stopped.
#
#   bash pipeline/refit.sh
cd "$(dirname "$0")/.."
# REFIT_ROOT redirects every output, for a dry run that must not touch the campaign root.
R="${REFIT_ROOT:-logs/thesis/fits}"
S=logs/thesis/pipeline.log
mkdir -p "$R"
GATE="$R/gate_dataset"
# GATE_CONFIG and INFERENCE_DIR let the gate-variant test reuse one inference pass.
GATE_CONFIG="${GATE_CONFIG:-config/sensor_gate.yaml}"
INF="${INFERENCE_DIR:-$R/detector_inference}"
# No ROS sourcing: install/setup.bash puts the ROS `experiments` package on PYTHONPATH,
# which shadows this repository's experiments/ directory and breaks the imports.
log() { echo "- $(date '+%F %T') step 5: $*" >> "$S"; echo "$*"; }
run() {  # run OUTPUT_DIR command...
  local out="$1"; shift
  if [ -e "$out" ]; then log "skip $(basename "$out") (exists)"; return 0; fi
  log "start $(basename "$out")"
  "$@" > "${out}.log" 2>&1 || { log "FAILED $(basename "$out"), see ${out}.log"; exit 1; }
  log "done $(basename "$out")"
}
run "$INF" python3 pipeline/detect.py \
  --output "$INF" --batch-size 8
run "$R/gate_dataset" python3 pipeline/gate.py \
  --inference "$INF" --gate "$GATE_CONFIG" --output "$GATE"
run "$R/correction" python3 pipeline/fit_correction.py \
  --gate-dataset "$GATE" --output "$R/correction"
run "$R/corrected_residuals" python3 pipeline/corrected_residuals.py \
  --correction "$R/correction" --output "$R/corrected_residuals"
run "$R/covariance" python3 pipeline/fit_covariance.py \
  --residuals "$R/corrected_residuals" --output "$R/covariance"
run "$R/ddev_evaluation" python3 pipeline/evaluate_ddev.py \
  --correction "$R/correction" --covariance "$R/covariance" --output "$R/ddev_evaluation"
run "$R/runtime_r012" python3 pipeline/package_runtime.py \
  --correction-manifest "$R/correction/manifest.json" \
  --covariance-models "$R/covariance/models.npz" \
  --world src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf --output "$R/runtime_r012"
run "$R/planning_precision" python3 pipeline/planning_precision.py \
  --covariance-models "$R/covariance/models.npz" \
  --output "$R/planning_precision"
log "ALL DONE"
