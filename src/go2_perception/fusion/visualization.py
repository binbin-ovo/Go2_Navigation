import cv2
from go2_perception.rgb.visualization import draw_detections
from .projection import project_points


def draw_fusion(image, rgb, lidar, calibration):
    canvas = draw_detections(image, rgb)
    for obstacle in lidar.obstacles:
        uv, _, _ = project_points(obstacle.points, calibration)
        for u, v in uv:
            cv2.circle(canvas, (int(u), int(v)), 2, (0, 100, 255), -1)
    cv2.putText(
        canvas,
        "REAL SENSOR FUSION NOT VALIDATED",
        (8, canvas.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
    )
    return canvas
