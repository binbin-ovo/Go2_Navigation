"""Headless XY views plus colored PLY usable in Open3D's interactive viewer."""

from pathlib import Path
import cv2
import numpy as np


def save_visualization(original, result, output):
    import open3d as o3d

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    colors = np.full((len(result.points), 3), 0.5)
    colors[result.labels == -2] = [0.2, 0.7, 0.2]
    palette = np.array([[1, 0.3, 0.1], [0.2, 0.5, 1], [1, 0.7, 0.1], [0.8, 0.2, 0.8]])
    for o in result.obstacles:
        colors[result.labels == o.cluster_id] = palette[o.cluster_id % len(palette)]
    if len(result.points):
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(result.points))
        cloud.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(str(out / "segmented.ply"), cloud)
    raw = np.asarray(original)[:, :3]
    raw = raw[np.isfinite(raw).all(axis=1)]
    bound = max(1.0, float(np.abs(raw[:, :2]).max())) if len(raw) else 1.0

    def pixel(p):
        return np.rint(
            np.column_stack((300 + p[:, 0] * 270 / bound, 300 - p[:, 1] * 270 / bound))
        ).astype(int)

    panels = []
    for title, p, c in [
        ("Original XY", raw, np.full((len(raw), 3), 0.65)),
        ("Ground green / noise gray", result.points, colors),
        ("Obstacle AABBs XY", result.points, colors),
    ]:
        canvas = np.full((600, 600, 3), 25, np.uint8)
        for xy, color in zip(pixel(p), c):
            cv2.circle(
                canvas, tuple(xy), 2, tuple(int(v * 255) for v in color[::-1]), -1
            )
        if title.startswith("Obstacle"):
            for obstacle in result.obstacles:
                corners = pixel(np.array([obstacle.bbox_min, obstacle.bbox_max]))
                cv2.rectangle(
                    canvas, tuple(corners[0]), tuple(corners[1]), (255, 255, 255), 1
                )
        cv2.putText(
            canvas, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )
        panels.append(canvas)
    cv2.imwrite(str(out / "lidar.png"), np.hstack(panels))
