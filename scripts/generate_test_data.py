from pathlib import Path
import argparse
import cv2
import numpy as np
import open3d as o3d
from go2_perception.io.synthetic import cases
from go2_perception.fusion.calibration import Calibration
from go2_perception.fusion.projection import project_points
from go2_perception.io.serialization import save_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/sample")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for name, points in cases().items():
        np.save(out / f"{name}.npy", points)
    points = cases()["two_obstacles"]
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    for suffix in ("pcd", "ply"):
        o3d.io.write_point_cloud(str(out / f"test.{suffix}"), cloud)
    np.column_stack((points, np.ones(len(points)))).astype("<f4").tofile(
        out / "test.bin"
    )
    calibration = Calibration.from_yaml("configs/synthetic_calibration.yaml")
    image = np.full((480, 640, 3), 220, np.uint8)
    detections = []
    for index, obstacle in enumerate((points[1600:1820], points[1820:])):
        uv, _, _ = project_points(obstacle, calibration)
        low = uv.min(0) - 2
        high = uv.max(0) + 2
        box = [float(low[0]), float(low[1]), float(high[0]), float(high[1])]
        cv2.rectangle(
            image, tuple(low.astype(int)), tuple(high.astype(int)), (80, 100, 180), -1
        )
        detections.append(
            dict(
                class_id=index,
                class_name=f"synthetic_fixture_{index}",
                confidence=1.0,
                bbox_xyxy=box,
            )
        )
    cv2.imwrite(str(out / "synthetic.jpg"), image)
    save_json(out / "synthetic_detections.json", detections)
    print(f"Generated deterministic synthetic data in {out}")
