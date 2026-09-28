#!/bin/bash
# Solve the 30 final offline routes (5 tasks x 6 conditions) and bind them into one
# execution config per seed. Skips any task whose routes already exist.
#
#   bash pipeline/routes.sh
cd "$(dirname "$0")/.."
R=logs/thesis
C=$R/campaign_configs
S=logs/thesis/pipeline.log
# The route solver imports the ROS experiments package (world profiles), so it needs ROS.
source /opt/ros/humble/setup.bash >/dev/null 2>&1
source install/setup.bash >/dev/null 2>&1
log() { echo "- $(date '+%F %T') step 7: $*" >> "$S"; echo "$*"; }
mkdir -p "$R/routes"
for task in thesis10_camera_a_western_dock_detour thesis10_camera_b_cross_warehouse_detour \
            thesis10_camera_c_inner_warehouse_detour thesis10_camera_e_eastern_detour \
            thesis10_camera_e_long_cross_warehouse_detour; do
  out="$R/routes/$task"
  if [ -e "$out" ]; then log "skip $task (exists)"; continue; fi
  log "solving $task"
  python3 pipeline/solve_routes.py \
    --campaign "$C/route_planning_campaign.yaml" --task "$task" --output "$out" \
    > "$out.log" 2>&1 || { log "FAILED $task, see $out.log"; exit 1; }
done
for seed in 91500 91501 91502; do
  cfg="$C/campaign_seed${seed}.yaml"
  [ -e "$cfg" ] && continue
  python3 pipeline/bind_routes.py \
    --campaign "$C/execution_template_seed${seed}.yaml" \
    --routes-root "$R/routes" --output "$cfg" \
    || { log "FAILED binding routes for seed $seed"; exit 1; }
done
log "ALL DONE: 30 routes solved and bound into three per-seed configs"
