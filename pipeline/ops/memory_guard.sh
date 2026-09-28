#!/bin/bash
# Restart the capture before it exhausts memory.
#
# capture_positions.py keeps every image hash in memory for deduplication, so its
# RSS grows with the row count: measured ~0.03 GB/min, from 0.2 GB at start to
# 3.2 GB after 14 minutes of writing. On this 15 GB machine it starts swapping
# around 3 GB, which halved the observed rate (9525 -> 4725 rows/h), and it would
# be OOM-killed long before the 51380-row target.
#
# A restart resets RSS to ~0.2 GB and costs one resume preflight (~6 min of
# hashing). That is far cheaper than swapping for hours or losing the run.
#
# The capture is resumable and the runner passes --resume automatically, so a
# restart never loses committed rows.
#   bash pipeline/ops/memory_guard.sh [LIMIT_GB] [LOG] [PROCESS]
# PROCESS is matched with a bracketed first letter so the guard never matches itself.
LIMIT_GB="${1:-5.5}"
LOG="${2:-logs/thesis/captures/v5/capture.log}"
PROCESS="${3:-capture_bbox_grid}"
PATTERN="[${PROCESS:0:1}]${PROCESS:1}"
while true; do
  pid=$(ps -eo pid,args | grep "$PATTERN" | grep python | awk '{print $1}' | head -1)
  if [ -n "$pid" ]; then
    rss_gb=$(ps -o rss= --pid "$pid" 2>/dev/null | awk '{printf "%.2f",$1/1048576}')
    over=$(awk -v a="${rss_gb:-0}" -v b="$LIMIT_GB" 'BEGIN{print (a>b)?1:0}')
    if [ "$over" = "1" ]; then
      echo "[$(date +%H:%M:%S)] MEMORY GUARD: capture at ${rss_gb} GB > ${LIMIT_GB} GB -- restarting" | tee -a "$LOG"
      # SIGTERM lets it finish the current batch and close the index cleanly.
      kill "$pid" 2>/dev/null
      sleep 20
      kill -9 "$pid" 2>/dev/null
      # the runner's retry loop brings the simulator back up and resumes
    fi
  fi
  sleep 60
done
