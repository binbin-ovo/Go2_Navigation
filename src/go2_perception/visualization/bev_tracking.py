"""Headless OpenCV BEV: x to the right, y up, all distances in metres."""

from pathlib import Path
import cv2
import numpy as np

COLORS = {
    "MOVING": (40, 170, 250),
    "STATIONARY": (120, 200, 60),
    "UNCERTAIN": (170, 170, 180),
}


def render_bev(
    tracks,
    timestamp,
    frame_id,
    robot_pose=None,
    bounds=(-2, 10, -4, 4),
    size=(1000, 750),
):
    xmin, xmax, ymin, ymax = bounds
    w, h = size
    if not np.isfinite(bounds).all() or xmin >= xmax or ymin >= ymax or min(w, h) < 300:
        raise ValueError("Invalid BEV bounds/size")
    canvas = np.full((h, w, 3), (27, 23, 20), np.uint8)
    margin = 70

    def xy(p):
        return tuple(
            np.rint(
                [
                    margin + (p[0] - xmin) / (xmax - xmin) * (w - 2 * margin),
                    h - margin - (p[1] - ymin) / (ymax - ymin) * (h - 2 * margin),
                ]
            ).astype(int)
        )

    for x in np.arange(np.ceil(xmin), xmax + 1):
        cv2.line(canvas, xy([x, ymin]), xy([x, ymax]), (55, 50, 45), 1)
        cv2.putText(
            canvas,
            f"{x:g}",
            (xy([x, ymin])[0] - 8, h - margin + 20),
            0,
            0.4,
            (175, 175, 175),
            1,
        )
    for y in np.arange(np.ceil(ymin), ymax + 1):
        cv2.line(canvas, xy([xmin, y]), xy([xmax, y]), (55, 50, 45), 1)
        cv2.putText(
            canvas,
            f"{y:g}",
            (margin - 32, xy([xmin, y])[1] + 5),
            0,
            0.4,
            (175, 175, 175),
            1,
        )
    cv2.putText(
        canvas,
        f"{frame_id} | t={timestamp:.2f}s | local odometry, not a global map"
        if not frame_id.startswith("fixed_sensor:")
        else f"{frame_id} | SENSOR-RELATIVE velocity only | t={timestamp:.2f}s",
        (20, 28),
        0,
        0.55,
        (240, 240, 240),
        1,
    )
    cv2.putText(
        canvas,
        "+x (m) ->    +y (m) up | body: x forward, y left",
        (margin, h - 18),
        0,
        0.5,
        (210, 210, 210),
        1,
    )
    if robot_pose is not None:
        p = robot_pose.T_parent_child[:3, 3]
        direction = robot_pose.T_parent_child[:2, 0]
        cv2.circle(canvas, xy(p), 7, (255, 200, 70), -1)
        cv2.arrowedLine(
            canvas,
            xy(p),
            xy(p[:2] + direction * 0.7),
            (255, 200, 70),
            2,
            tipLength=0.25,
        )
        cv2.putText(
            canvas, "robot", xy(p[:2] + [0.1, -0.3]), 0, 0.45, (255, 200, 70), 1
        )
    elif frame_id.startswith("fixed_sensor:"):
        cv2.drawMarker(canvas, xy([0, 0]), (255, 200, 70), cv2.MARKER_CROSS, 14, 2)
        cv2.putText(canvas, "fixed sensor", xy([0.1, -0.3]), 0, 0.45, (255, 200, 70), 1)
    for t in tracks:
        color = COLORS[t.motion_state]
        p = t.position
        history = [xy(v["position"]) for v in t.history]
        if len(history) > 1:
            cv2.polylines(
                canvas,
                [np.array(history, np.int32)],
                False,
                tuple(int(c * 0.55) for c in color),
                2,
            )
        future = [xy(p)] + [xy(v.position) for v in t.predicted_positions]
        for i in range(1, len(future)):
            cv2.line(canvas, future[i - 1], future[i], color, 1, cv2.LINE_AA)
        if t.bbox_min is not None:
            cv2.rectangle(canvas, xy(t.bbox_min), xy(t.bbox_max), color, 1)
        cv2.circle(canvas, xy(p), 5, color, -1)
        cv2.arrowedLine(canvas, xy(p), xy(p[:2] + t.velocity), color, 2, tipLength=0.25)
        label = f"#{t.track_id} {t.status} {np.linalg.norm(t.velocity):.2f} m/s"
        cv2.putText(canvas, label, (xy(p)[0] + 9, xy(p)[1] - 10), 0, 0.42, color, 1)
    for index, (label, color) in enumerate(COLORS.items()):
        cv2.putText(canvas, label, (20 + index * 180, 52), 0, 0.45, color, 1)
    return canvas


def save_bev(path, *args, **kwargs):
    image = render_bev(*args, **kwargs)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Cannot write {path}")
    return image
