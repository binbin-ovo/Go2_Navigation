# Optional ROS 2 adapter boundary
No ROS imports or fabricated Go2 topics are included. Claude must verify topics, encoding,
PointCloud2 field offsets, timestamps, frames and QoS before implementing a subscriber.
Convert bgr8 images and XYZ/XYZI arrays to `RGBMessage` / `LidarMessage`, then call
`process_rgb` / `process_lidar`. See `docs/SENSOR_INTERFACE.md`.
A sensor_bridge.py placeholder is deliberately omitted until transport details are verified.
