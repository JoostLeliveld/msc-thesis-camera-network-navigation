#!/bin/bash
# Stop the capture cleanly if free disk falls below a floor.
#
# The projected need (~24 GB after deduplication) fits the ~28.7 GB free, but the
# margin is under 1 GB and the dedup rate could worsen later in the run. A full
# disk mid-write corrupts the capture; a clean stop leaves a resumable prefix
# that passes the 29-check preflight. So this watches and stops rather than
# hoping.
FLOOR_GB="${1:-3}"
LOG=logs/thesis/captures/v5/capture.log
while true; do
  free_gb=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
  if [ "${free_gb:-99}" -lt "$FLOOR_GB" ]; then
    echo "[$(date +%H:%M:%S)] WATCHDOG: free ${free_gb}GB < ${FLOOR_GB}GB -- stopping capture cleanly" | tee -a "$LOG"
    for p in $(ps -eo pid,args | grep -E "run_recapture_v5|capture_bbox_grid" | grep -v grep | awk '{print $1}'); do
      kill "$p" 2>/dev/null
    done
    sleep 10
    for p in $(ps -eo pid,args | grep -E "ign gazebo|parameter_bridge" | grep -v grep | awk '{print $1}'); do
      kill -9 "$p" 2>/dev/null
    done
    exit 1
  fi
  sleep 60
done
