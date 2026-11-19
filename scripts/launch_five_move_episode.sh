#!/bin/bash
# Local preparation or an explicitly supervised area episode under one command owner.
set -eu
mode=${1:-}
scope=operator_inspected_static_disc
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in --prepare-only|--live-exclusive) ;; *) echo "Usage: bash scripts/launch_five_move_episode.sh --prepare-only|--live-exclusive [--move-budget N --episode-seconds S --area-radius-m M]" >&2; exit 2;; esac
if [[ "$mode" == --live-exclusive && "${GO2_SUPERVISED_LIVE:-0}" != 1 ]]; then
  echo "Live launch requires GO2_SUPERVISED_LIVE=1 and exclusive command ownership." >&2
  exit 2
fi
cd "$(dirname "$0")/.."
# One local workspace outside cloud-synchronized folders; see docs/MAC_CONNECTION.md.
workspace="${GO2_WORKSPACE:-$HOME/Robotics/Go2}"
export GO2_WORKSPACE="$workspace"
control="$workspace/code/control"
export GO2_HARDWARE=0 MUJOCO_RENDER_TEST=0 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:scripts
mkdir -p "$workspace/runs/five_move"
trial_dir=$(mktemp -d "$workspace/runs/five_move/trial-XXXXXX")
core="$workspace/envs/core/bin/python"
"$core" hardware/make_capability.py --profile go2-effective-v1 --turn-scope "$scope" \
  --heading-file "$workspace/evidence/hw_readiness/camera_heading.json" --output "$trial_dir/capability.json"
echo "Trial directory: $trial_dir"
if [[ "$mode" == --prepare-only ]]; then exit 0; fi
exec "$core" scripts/run_vln_supervised.py --five-move-episode --navigation-mode live --execute-live --exclusive-command-source \
  --profile go2-effective-v1 --capability "$trial_dir/capability.json" --control-root "$control" \
  --hardware-python "$workspace/envs/hw/bin/python" --unitree-sdk "$workspace/sdk/unitree_sdk2_python" \
  --camera go2 --lidar go2 --iface en9 --camera-host-ip 192.168.123.222 --backend lightnav \
  --model-python "$workspace/envs/model/bin/python" --assets "$workspace/weights/lightnav" \
  --runtime-config configs/pretrained_vln.yaml --device mps --calibration configs/go2_front_camera_calibration.yaml \
  --camera-latency configured --preparation-mode cached --tensor-evidence skip --min-free-disk-gb 10 \
  --output "$trial_dir/session" --port 8766 "$@"
