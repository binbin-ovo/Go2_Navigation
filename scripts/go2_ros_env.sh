# source this: uses the ROS 2 Humble libs bundled with Isaac Sim, bound to the Go2 wired interface.
export GO2_PY=/home/yushan/anaconda3/envs/env_isaaclab/bin/python
_B=/home/yushan/anaconda3/envs/env_isaaclab/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/humble
export PYTHONPATH=$_B/rclpy:$PYTHONPATH
export LD_LIBRARY_PATH=$_B/lib:$LD_LIBRARY_PATH
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="eno2" priority="default" multicast="default"/></Interfaces></General></Domain></CycloneDDS>'
