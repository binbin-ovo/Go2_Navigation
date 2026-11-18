#!/bin/bash
# Claude's opt-in one-cycle launcher. Prepare-only performs local file checks.
set -eu
mode=${1:-}
scope=${2:-sensor_only}
case "$mode" in --prepare-only|--live-exclusive) ;; *) echo "Usage: bash scripts/launch_effective_trial.sh --prepare-only|--live-exclusive [sensor_only|operator_inspected_static_disc]" >&2; exit 2;; esac
if [[ "$mode" == --live-exclusive && "${GO2_SUPERVISED_LIVE:-0}" != 1 ]]; then
  echo "Live launch requires GO2_SUPERVISED_LIVE=1 and exclusive command ownership." >&2
  exit 2
fi
cd "$(dirname "$0")/.."
base="$HOME/Desktop/Go2_Navigation"
ready="$HOME/Desktop/Go2_Navigation-hw-readiness"
control="$HOME/Desktop/Go2_Navigation-go2-control"
export GO2_HARDWARE=0 MUJOCO_RENDER_TEST=0 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:scripts
mkdir -p data/outputs/effective_trials
trial_dir=$(mktemp -d "$PWD/data/outputs/effective_trials/trial-XXXXXX")
core="$ready/.venv/bin/python"
"$core" hardware/make_capability.py --profile go2-effective-v1 --turn-scope "$scope" \
  --heading-file "$ready/data/outputs/hw_readiness/camera_heading.json" --output "$trial_dir/capability.json"
echo "Trial directory: $trial_dir"
if [[ "$mode" == --prepare-only ]]; then exit 0; fi
exec "$core" scripts/run_vln_supervised.py --navigation-mode live --execute-live --exclusive-command-source \
  --profile go2-effective-v1 --capability "$trial_dir/capability.json" --control-root "$control" \
  --hardware-python "$base/.venv-go2/bin/python" --unitree-sdk "$base/third_party/unitree_sdk2_python" \
  --camera go2 --lidar go2 --iface en9 --camera-host-ip 192.168.123.222 --backend lightnav \
  --model-python "$base/data/outputs/phase6c/model-env/bin/python" --assets "$base/data/outputs/phase6c/lightnav" \
  --runtime-config configs/pretrained_vln.yaml --device mps --calibration configs/go2_front_camera_calibration.yaml \
  --camera-latency configured --preparation-mode cached --tensor-evidence skip --min-free-disk-gb 10 \
  --output "$trial_dir/session" --port 8766
