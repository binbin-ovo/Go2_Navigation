#!/usr/bin/env bash
# Installs Unitree's Python SDK into the existing "go2" conda env, then checks the imports.
set -euo pipefail

cd "$(dirname "$0")/.."
REPO=https://github.com/unitreerobotics/unitree_sdk2_python
DEST=third_party/unitree_sdk2_python
PY=$HOME/anaconda3/envs/go2/bin/python

if [ ! -d "$DEST" ]; then
    git clone --depth 1 "$REPO" "$DEST"
fi
git -C "$DEST" log -1 --format='SDK commit: %h %ad' --date=short

"$PY" -m pip install -e "$DEST"
"$PY" -m pip list 2>/dev/null | grep -iE 'cyclonedds|unitree|opencv|numpy'

"$PY" -c "
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LidarState_
print('SDK imports OK')
"
