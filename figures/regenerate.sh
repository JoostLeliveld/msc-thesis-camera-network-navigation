#!/usr/bin/env bash
# Generate all thesis figures and table rows from a complete evidence bundle.
set -eo pipefail
cd "$(dirname "$0")/.."
python3 figures/make_runtime_coverage.py
for script in figures/make_*.py; do
  [[ "$script" == figures/make_removal_mechanism.py || "$script" == figures/make_runtime_coverage.py ]] && continue
  python3 "$script"
done
for task in thesis10_camera_a_western_dock_detour thesis10_camera_b_cross_warehouse_detour \
            thesis10_camera_c_inner_warehouse_detour thesis10_camera_e_eastern_detour \
            thesis10_camera_e_long_cross_warehouse_detour; do
  python3 figures/make_removal_mechanism.py "$task"
done
