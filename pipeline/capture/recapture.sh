#!/bin/bash
# Bring warehouse_v2 up headless in an isolated transport partition and capture a
# predeclared pose file (v9: the fresh spatially balanced audit). A presence watchdog kills
# the capture if the robot vanishes from the simulator, so the retry loop restarts it.
#
# capture_positions.py refuses unisolated transport for training-grade capture,
# so ROS_LOCALHOST_ONLY / IGN_IP / GZ_IP / ROS_DOMAIN_ID / IGN_PARTITION are all
# set here and exported to BOTH the simulator and the capture, or they will not
# see each other.
#
# The capture is resumable: it appends to capture_index.csv and skips poses that
# already have a complete five-camera batch. This script therefore retries the
# whole capture a few times, which recovers from a mid-run simulator stall
# without discarding completed work.
#
#   bash pipeline/capture/recapture.sh POSE_FILE OUT_DIR
# NOTE: no `set -u`. ROS's setup.bash dereferences unset variables, which
# terminates the shell under `set -u` before anything is logged.

cd "$(dirname "$0")/../.."
REPO="$(pwd)"
POSES="$1"
OUT="$2"
[ -f "$POSES" ] && [ -n "$OUT" ] || { echo "usage: recapture.sh POSE_FILE OUT_DIR" >&2; exit 2; }
LOGDIR="${REPO}/$(dirname "$OUT")"
# The segmentation overlay and the dedup shim were built for v5 and are reused as is.
ASSETS="${REPO}/logs/thesis/captures/v5"
mkdir -p "$LOGDIR"
SIMLOG="${LOGDIR}/sim.log"
CAPLOG="${LOGDIR}/capture.log"

export ROS_LOCALHOST_ONLY=1
export IGN_IP=127.0.0.1
export GZ_IP=127.0.0.1
export ROS_DOMAIN_ID=77
export IGN_PARTITION=recapture_v9
export OMP_NUM_THREADS=2

source /opt/ros/humble/setup.bash >/dev/null 2>&1
source install/setup.bash >/dev/null 2>&1
export PYTHONPATH="${REPO}/src/perception:${PYTHONPATH:-}"


# CAPTURE-ONLY camera models: identical to the installed ones plus a 1 Hz
# semantic segmentation sensor. Commit 8449efd3 removed that sensor because it
# was the Gazebo bottleneck at 5 Hz navigation, but the master capture needs the
# robot mask as ground truth for reference_class, the stage-04 label contract
# and an honest q target. The capture is static -- teleport, settle, grab one
# frame -- so the render cost does not apply.
#
# The overlay lives under logs/, NOT src/ or install/, and is prepended to the
# resource path ONLY by this script. No navigation or campaign launch can load
# it, and nothing tracked by git changes.
CAPTURE_MODELS="${ASSETS}/capture_models"
INSTALLED_MODELS="${REPO}/install/sim/share/sim/models"
if [ ! -d "$CAPTURE_MODELS" ]; then
  echo "ABORT: ${CAPTURE_MODELS} missing; rebuild the capture-only model overlay" >&2
  exit 2
fi
# gazebo.launch.py builds GZ_SIM_RESOURCE_PATH with SetEnvironmentVariable, which
# REPLACES whatever this shell exports, so prepending a directory here has no
# effect. The models are therefore swapped in place inside install/, which is
# gitignored build output regenerated from src/ by colcon. The originals are
# saved and restored on exit, so a later navigation run gets the fast
# no-segmentation models back even if this script is killed.
CAPTURE_BACKUP="${LOGDIR}/installed_models_backup"
restore_models() {
  if [ -d "$CAPTURE_BACKUP" ]; then
    for m in "$CAPTURE_BACKUP"/*/; do
      n=$(basename "$m")
      cp -f "${m}model.sdf" "${INSTALLED_MODELS}/${n}/model.sdf" 2>/dev/null
    done
    echo "[$(date +%H:%M:%S)] restored the installed (no-segmentation) camera models" | tee -a "$CAPLOG"
  fi
}
install_capture_models() {
  mkdir -p "$CAPTURE_BACKUP"
  for m in "$CAPTURE_MODELS"/*/; do
    n=$(basename "$m")
    if [ ! -f "${CAPTURE_BACKUP}/${n}/model.sdf" ]; then
      mkdir -p "${CAPTURE_BACKUP}/${n}"
      cp -f "${INSTALLED_MODELS}/${n}/model.sdf" "${CAPTURE_BACKUP}/${n}/model.sdf"
    fi
    cp -f "${m}model.sdf" "${INSTALLED_MODELS}/${n}/model.sdf"
  done
  echo "[$(date +%H:%M:%S)] installed capture-only camera models (with segmentation)" | tee -a "$CAPLOG"
}

sim_pid=""
teardown() {
  [ -n "$sim_pid" ] && kill "$sim_pid" >/dev/null 2>&1
  for pat in "ign gazebo" "gz sim" "ros_gz_bridge/parameter_bridge" \
             "sim/lib/sim/clock_throttle_node" "robot_state_publisher"; do
    pkill -9 -f "$pat" >/dev/null 2>&1
  done
  sleep 3
}
on_exit() { teardown; restore_models; }
trap on_exit EXIT

bring_up_sim() {
  # A stale simulator from a previous attempt would answer on the same partition
  # and silently serve the OLD world, so always start from a clean slate.
  teardown
  ros2 daemon stop >/dev/null 2>&1
  echo "[$(date +%H:%M:%S)] launching simulator" | tee -a "$CAPLOG"
  # The launch and the capture MUST share the isolation variables. This shell
  # inherits ROS_LOCALHOST_ONLY=0 from the environment, and sourcing ROS does not
  # clear it, so both are passed explicitly with `env` rather than relying on
  # export alone.
  env ROS_LOCALHOST_ONLY=1 IGN_IP=127.0.0.1 GZ_IP=127.0.0.1 \
      ROS_DOMAIN_ID="$ROS_DOMAIN_ID" IGN_PARTITION="$IGN_PARTITION" \
      IGN_GAZEBO_RESOURCE_PATH="$IGN_GAZEBO_RESOURCE_PATH" \
      GZ_SIM_RESOURCE_PATH="$GZ_SIM_RESOURCE_PATH" \
  ros2 launch sim bringup_sim.launch.py \
    world:=warehouse_v2.world.sdf world_name:=warehouse_v2 \
    headless:=true show_pose_markers:=false reset_world:=false \
    spawn_x:=0.0 spawn_y:=-5.0 spawn_z:=0.05 spawn_yaw:=0.0 \
    use_lidar:=false bridge_scan:=false bridge_contacts:=false \
    bridge_camera_a:=true bridge_camera_b:=true bridge_camera_c:=true \
    bridge_camera_d:=true bridge_camera_e:=true \
    bridge_segmentation:=true bridge_segmentation_b:=true \
    bridge_segmentation_c:=true bridge_segmentation_d:=true \
    bridge_segmentation_e:=true \
    > "$SIMLOG" 2>&1 &
  sim_pid=$!
  for i in $(seq 1 40); do
    n=0
    for t in /external_camera/image_raw /external_camera_b/image_raw \
             /external_camera_c/image_raw /external_camera_d/image_raw \
             /external_camera_e/image_raw \
             /external_camera/segmentation/labels_map \
             /external_camera_b/segmentation/labels_map \
             /external_camera_c/segmentation/labels_map \
             /external_camera_d/segmentation/labels_map \
             /external_camera_e/segmentation/labels_map; do
      # --no-daemon: the ros2 daemon caches discovery state from earlier,
      # differently-configured runs and then answers every query with
      # "!rclpy.ok()", which looks exactly like "no streams" while Gazebo and
      # the bridges are publishing normally.
      # The message type is given explicitly: without the daemon, `echo` otherwise races
      # discovery and fails with "Could not determine the type", so under load ten checks
      # in a row almost never all passed although every stream was up.
      env ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID="$ROS_DOMAIN_ID" \
        timeout 8 ros2 topic echo --once --no-daemon "$t" sensor_msgs/msg/Image >/dev/null 2>&1 && n=$((n+1))
    done
    if [ "$n" = "10" ]; then
      echo "[$(date +%H:%M:%S)] all five RGB + five label streams up" | tee -a "$CAPLOG"
      return 0
    fi
    sleep 4
  done
  echo "[$(date +%H:%M:%S)] ABORT: streams did not come up ($n/10)" | tee -a "$CAPLOG"
  tail -20 "$SIMLOG" | tee -a "$CAPLOG"
  return 1
}

rows_done() {
  [ -f "${OUT}/capture_index.csv" ] && echo $(( $(wc -l < "${OUT}/capture_index.csv") - 1 )) || echo 0
}

install_capture_models
TARGET_ROWS=$(( $(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$POSES") * 5 ))
# 60 attempts, not 12: the memory guard restarts the capture roughly every 20
# minutes (RSS grows ~0.03 GB/min from the in-memory dedup hash set), so a full
# run needs many more resumes than a failure-only retry budget would allow.
for attempt in $(seq 1 60); do
  have=$(rows_done)
  echo "[$(date +%H:%M:%S)] attempt ${attempt}: ${have}/${TARGET_ROWS} rows present" | tee -a "$CAPLOG"
  if [ "$have" -ge "$TARGET_ROWS" ]; then
    echo "[$(date +%H:%M:%S)] capture already complete" | tee -a "$CAPLOG"
    break
  fi
  # capture_positions.py refuses to write into an existing --out unless
  # --resume is given; it then AUDITS the prefix (29 integrity checks) before
  # appending. Without this every retry died on FileExistsError.
  if [ -f "${OUT}/capture_index.csv" ]; then RESUME_FLAG="--resume"; else RESUME_FLAG=""; fi
  bring_up_sim || { sleep 20; continue; }
  python3 pipeline/capture/presence_watchdog.py "$OUT" >> "${LOGDIR}/presence_watchdog.log" 2>&1 &
  watchdog_pid=$!
  # CAPTURE_DEDUP_FACTOR: the preflight sizes the remaining rows at the largest
  # observed PNG with NO deduplication. This capture stores one file per unique
  # image hash and 67% of rows reuse an identical frame, so that estimate is ~4x
  # the measured cost. The factor is applied through a sitecustomize shim rather
  # than by editing audit_capture_resume.py, whose hash the capture verifies.
  env ROS_LOCALHOST_ONLY=1 IGN_IP=127.0.0.1 GZ_IP=127.0.0.1 \
      ROS_DOMAIN_ID="$ROS_DOMAIN_ID" IGN_PARTITION="$IGN_PARTITION" \
      PYTHONPATH="${ASSETS}/shim:${PYTHONPATH}" \
  python3 pipeline/capture/capture_positions.py \
    --world warehouse_v2.world.sdf \
    --pose-file "$POSES" \
    --pose-validity footprint \
    --with-semantic \
    --settle-s 0.40 \
    --min-new-rgb-frames 2 \
    $RESUME_FLAG \
    --out "$OUT" \
    >> "$CAPLOG" 2>&1
  status=$?
  kill "$watchdog_pid" >/dev/null 2>&1
  echo "[$(date +%H:%M:%S)] capture exited ${status}, rows now $(rows_done)" | tee -a "$CAPLOG"
  if [ "$status" = "0" ]; then break; fi
  teardown
  sleep 15
done

echo "[$(date +%H:%M:%S)] FINAL rows: $(rows_done)/${TARGET_ROWS}" | tee -a "$CAPLOG"
