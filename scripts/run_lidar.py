from pathlib import Path
import argparse
from go2_perception.io.pointcloud_loader import load_pointcloud
from go2_perception.io.config import load_config
from go2_perception.io.serialization import save_json
from go2_perception.lidar.processor import LidarProcessor
from go2_perception.lidar.visualization import save_visualization

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config", default="configs/lidar.yaml")
    p.add_argument("--bin-fields", type=int, choices=[3, 4], default=4)
    a = p.parse_args()
    points = load_pointcloud(a.input, a.bin_fields)
    result = LidarProcessor(**load_config(a.config)).process(points)
    save_json(Path(a.output) / "obstacles.json", result)
    save_visualization(points, result, a.output)
    print(
        f"{len(result.ground)} ground; {len(result.obstacles)} clusters; {len(result.unclassified)} unclassified; {result.latency_ms:.2f} ms"
    )
