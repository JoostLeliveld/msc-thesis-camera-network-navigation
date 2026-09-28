#!/bin/bash
# Hold a systemd inhibitor so a closed lid does not suspend the machine while
# the master capture runs.
#
# HandleLidSwitch defaults to `suspend` here (verified: gsettings reports
# 'suspend' on both AC and battery), so closing the laptop would freeze Gazebo,
# the capture and the guards mid-write. This blocks ONLY the lid/sleep handling
# and only for as long as this script runs -- it changes no system setting, so
# normal behaviour returns the moment it exits.
#
# It exits by itself once the capture is finished, so the laptop can sleep again
# without anyone remembering to clean up.
TARGET_ROWS="${1:-51380}"
INDEX=logs/thesis/captures/v5/part1/capture_index.csv
LOG=logs/thesis/captures/v5/capture.log

echo "[$(date +%H:%M:%S)] lid-suspend inhibited until ${TARGET_ROWS} rows" | tee -a "$LOG"
systemd-inhibit \
  --what=handle-lid-switch:sleep:idle \
  --who="master capture v5" \
  --why="warehouse_v2 recapture in progress" \
  --mode=block \
  bash -c '
    while true; do
      rows=$(( $(wc -l < "'"$INDEX"'" 2>/dev/null || echo 1) - 1 ))
      [ "$rows" -ge "'"$TARGET_ROWS"'" ] && break
      # also stop inhibiting if the capture is no longer running at all
      pgrep -f "run_recapture_v5" >/dev/null || { sleep 120; pgrep -f "run_recapture_v5" >/dev/null || break; }
      sleep 60
    done'
echo "[$(date +%H:%M:%S)] lid-suspend inhibitor released" | tee -a "$LOG"
