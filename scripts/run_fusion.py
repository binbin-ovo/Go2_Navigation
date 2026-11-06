from pathlib import Path
import argparse
import json
import cv2
from go2_perception.core.types import Detection2D, RGBResult
from go2_perception.io.image_loader import load_image
from go2_perception.io.pointcloud_loader import load_pointcloud
from go2_perception.io.config import load_config
from go2_perception.io.serialization import save_json
from go2_perception.rgb.detector import RGBDetector
from go2_perception.lidar.processor import LidarProcessor
from go2_perception.fusion.calibration import Calibration
from go2_perception.fusion.association import FusionProcessor
from go2_perception.fusion.visualization import draw_fusion

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--points", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--detections", help="JSON Detection2D list; optional test fixture")
    p.add_argument("--rgb-timestamp", type=float, required=True)
    p.add_argument("--lidar-timestamp", type=float, required=True)
    p.add_argument("--clock-domain", required=True)
    a = p.parse_args()
    config = load_config(a.config)
    cal = Calibration.from_yaml(a.calibration)
    image = load_image(a.image)
    if a.detections:
        rgb = RGBResult(
            [Detection2D(**d) for d in json.loads(Path(a.detections).read_text())],
            (image.shape[1], image.shape[0]),
        )
    else:
        rgb = RGBDetector(**load_config(config["rgb"])).detect(image)
    rgb.timestamp = a.rgb_timestamp
    rgb.frame_id = cal.camera_frame
    rgb.clock_domain = a.clock_domain
    lidar = LidarProcessor(**load_config(config["lidar"])).process(
        load_pointcloud(a.points),
        timestamp=a.lidar_timestamp,
        frame_id=cal.lidar_frame,
        clock_domain=a.clock_domain,
    )
    fused = FusionProcessor(**load_config(config["fusion"])).fuse(rgb, lidar, cal)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "fusion.json", fused)
    cv2.imwrite(str(out / "fusion.jpg"), draw_fusion(image, rgb, lidar, cal))
    print(f"{len(fused.detections)} associations; {fused.validation}")
