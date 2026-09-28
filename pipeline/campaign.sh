#!/bin/bash
# Execute the final five-task dropout campaign seed by seed: 30 runs per seed, seeds 91500,
# 91501 and 91502 in that order, one simulator at a time. --resume skips runs already recorded
# in a seed's campaign_log.json, so the script can be restarted safely.
#
#   bash pipeline/campaign.sh [SEED ...]
cd "$(dirname "$0")/.."
R=logs/thesis/final_campaign
S=$R/STATUS
MIN_FREE_KB=6291456
source /opt/ros/humble/setup.bash >/dev/null 2>&1
source install/setup.bash >/dev/null 2>&1
log() { echo "- $(date '+%F %T') campaign: $*" >> "$S"; echo "$*"; }
mkdir -p "$R"

# Refuse stale code/configuration and a disk-full failure halfway through a seed.
python3 pipeline/campaign_manifest.py --campaign-root "$R" || exit 1
free_kb=$(df -Pk "$R" | awk 'NR == 2 {print $4}')
if [ "$free_kb" -lt "$MIN_FREE_KB" ]; then
  log "REFUSED: need at least 6 GiB free, found $((free_kb / 1024)) MiB"
  exit 1
fi
SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(91500 91501 91502)
for seed in "${SEEDS[@]}"; do
  free_kb=$(df -Pk "$R" | awk 'NR == 2 {print $4}')
  if [ "$free_kb" -lt "$MIN_FREE_KB" ]; then
    log "REFUSED before seed $seed: need at least 6 GiB free, found $((free_kb / 1024)) MiB"
    exit 1
  fi
  cfg="$R/campaign_configs/campaign_seed${seed}.yaml"
  [ -f "$cfg" ] || { log "FAILED: missing $cfg"; exit 1; }
  log "seed $seed started"
  python3 pipeline/campaign_runner.py \
    --config "$cfg" --log-root "$R/campaign/seed${seed}" --resume \
    >> "$R/campaign_seed${seed}.log" 2>&1
  rc=$?
  log "seed $seed finished, exit $rc"
  [ "$rc" = "0" ] || exit "$rc"
  CAMPAIGN_ROOT="$R/campaign" bash pipeline/ops/compress_finished_runs.sh --once || exit 1
  python3 pipeline/campaign_manifest.py --campaign-root "$R" || exit 1
done
log "ALL SEEDS DONE"
