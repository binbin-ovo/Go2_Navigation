"""Deterministic measured-failure-inspired fixtures, not replicas of private sensor data."""

import numpy as np
from scipy.spatial.transform import Rotation
from .synthetic import generate_sequence

SCENARIOS = (
    "fragmentation",
    "nearby_objects",
    "occlusion",
    "crossing",
    "rotation_wrap",
    "irregular",
    "outliers",
    "odometry_gap",
    "brief_stop",
    "stationary_noise",
)


def scenarios(seed=36):
    rng = np.random.default_rng(seed)
    result = {}
    for name in SCENARIOS:
        if name in ("crossing", "irregular"):
            result[name] = generate_sequence(name, seed)
            continue
        rows = []
        for i in range(100):
            t = i * 0.065
            speed = 0.0 if name not in ("occlusion", "brief_stop") else 0.6
            x = 3.0 + speed * t
            if name == "brief_stop":
                x = 3.0 + 0.6 * min(t, 2) + 0.6 * max(0, t - 4)
                speed = 0.6 if t < 2 or t >= 4 else 0.0
            centers = [np.array([x, 0, 0.6])]
            if name == "nearby_objects":
                centers = [np.array([3, -0.35, 0.6]), np.array([3, 0.35, 0.6])]
            truths = []
            detections = []
            for j, p in enumerate(centers):
                truths.append(
                    dict(
                        object_id=str(j),
                        position=p.tolist(),
                        velocity=[speed, 0],
                        frame_id="odom",
                    )
                )
                if name == "occlusion" and 30 <= i < 46:
                    continue  # 1.04 s, beyond 10 frames
                noisy = p + rng.normal(
                    0, 0.025 if name != "stationary_noise" else 0.045, 3
                )
                size = np.array([0.3, 0.3, 0.5])
                parts = [(noisy, size)]
                if name == "fragmentation" and i >= 15:
                    parts = [
                        (noisy + [0, -0.17, 0], np.array([0.3, 0.13, 0.5])),
                        (noisy + [0, 0.17, 0], np.array([0.3, 0.13, 0.5])),
                    ]
                for ppart, extent in parts:
                    detections.append(
                        dict(
                            detection_id=str(len(detections)),
                            position=ppart.tolist(),
                            bbox_min=(ppart - extent).tolist(),
                            bbox_max=(ppart + extent).tolist(),
                        )
                    )
            if name == "outliers" and i % 13 == 0:
                detections.append(
                    dict(
                        detection_id="outlier",
                        position=[8, float(rng.uniform(-3, 3)), 0.6],
                    )
                )
            pose = None
            frame = "odom"
            if name == "rotation_wrap":
                matrix = np.eye(4)
                matrix[:3, :3] = Rotation.from_euler("z", 2.8 + t * 0.15).as_matrix()
                matrix[:3, 3] = [0.05 * t, 0, 0]
                frame = "base_link"
                for d in detections:
                    p = (np.array(d["position"]) - matrix[:3, 3]) @ matrix[:3, :3]
                    d.update(
                        position=p.tolist(),
                        bbox_min=(p - 0.3).tolist(),
                        bbox_max=(p + 0.3).tolist(),
                    )
                pose = dict(
                    timestamp=t,
                    T_parent_child=matrix.tolist(),
                    parent_frame="odom",
                    child_frame="base_link",
                    clock_domain="synthetic",
                )
            # Posed replay resumes after a >.5s outage: both implementations must reset.
            if name == "odometry_gap" and 30 <= i < 42:
                continue
            rows.append(
                dict(
                    timestamp=t,
                    frame_id=frame,
                    clock_domain="synthetic",
                    detections=detections,
                    robot_pose=pose,
                    ground_truth=truths,
                    reset_before=name == "odometry_gap" and i == 42,
                )
            )
        result[name] = rows
    return result
