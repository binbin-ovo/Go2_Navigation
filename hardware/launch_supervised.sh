#!/bin/bash
# Shared by the two explicit operator launchers. Preparation only reads local evidence.
set -eu
kind=$1
shift
mode=${1:-}
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in
  --prepare-only|--live-exclusive) ;;
  *) echo "Expected --prepare-only or --live-exclusive" >&2; exit 2 ;;
esac
if [[ "$mode" == --live-exclusive && "${GO2_SUPERVISED_LIVE:-0}" != 1 ]]; then
  echo "Live launch requires GO2_SUPERVISED_LIVE=1 and exclusive command ownership." >&2
  exit 2
fi
runner=(scripts/run_vln_supervised.py)
case "$kind" in
  five-move) run_kind=five_move; scope=operator_inspected_static_disc; runner+=(--five-move-episode) ;;
  one-action) run_kind=effective_trials; scope=${1:-sensor_only}; set -- ;;
  *) echo "Unknown supervised strategy" >&2; exit 2 ;;
esac
cd "$(dirname "$0")/.."
workspace="${GO2_WORKSPACE:-$HOME/Robotics/Go2}"
export GO2_WORKSPACE="$workspace"
control="$workspace/code/control"
export GO2_HARDWARE=0 MUJOCO_RENDER_TEST=0 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:scripts
mkdir -p "$workspace/runs/$run_kind"
trial_dir=$(mktemp -d "$workspace/runs/$run_kind/trial-XXXXXX")
core="$workspace/envs/core/bin/python"
"$core" hardware/make_capability.py --profile go2-effective-v1 --turn-scope "$scope" \
  --heading-file "$workspace/evidence/hw_readiness/camera_heading.json" --output "$trial_dir/capability.json"
echo "Trial directory: $trial_dir"
if [[ "$mode" == --prepare-only ]]; then exit 0; fi
exec "$core" "${runner[@]}" --navigation-mode live --execute-live --exclusive-command-source \
  --profile go2-effective-v1 --capability "$trial_dir/capability.json" --control-root "$control" \
  --hardware-python "$workspace/envs/hw/bin/python" --unitree-sdk "$workspace/sdk/unitree_sdk2_python" \
  --camera go2 --lidar go2 --iface en9 --camera-host-ip 192.168.123.222 --backend lightnav \
  --model-python "$workspace/envs/model/bin/python" --assets "$workspace/weights/lightnav" \
  --runtime-config configs/pretrained_vln.yaml --device mps --calibration configs/go2_front_camera_calibration.yaml \
  --camera-latency configured --preparation-mode cached --tensor-evidence skip --min-free-disk-gb 10 \
  --output "$trial_dir/session" --port 8766 "$@"
