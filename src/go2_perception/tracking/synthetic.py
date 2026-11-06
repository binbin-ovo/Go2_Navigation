"""Eight deterministic noisy scenarios. Ground truth is odometry-frame, explicitly synthetic."""

import numpy as np
from scipy.spatial.transform import Rotation
from go2_perception.core.transforms import transform_points
from go2_perception.io.serialization import jsonable

SCENARIOS = (
    "stationary",
    "ego_motion",
    "moving",
    "crossing",
    "occlusion",
    "irregular",
    "noisy",
    "sparse",
)


def generate_sequence(scenario, seed=13):
    if scenario not in SCENARIOS:
        raise ValueError("Unknown scenario")
    rng = np.random.default_rng(seed)
    times = (
        np.r_[0, np.cumsum(rng.uniform(0.06, 0.2, 79))]
        if scenario == "irregular"
        else np.arange(80) * 0.1
    )
    sigma = 0.15 if scenario == "noisy" else 0.035
    rows = []
    for index, t in enumerate(times):
        pose = np.eye(4)
        if scenario == "ego_motion":
            pose[:3, 3] = [0.4 * t, 0.2 * np.sin(t), 0.3]
            pose[:3, :3] = Rotation.from_euler("z", 0.2 * t).as_matrix()
        states = [("object_1", np.array([4.0, 1.0, 0.6]), np.array([0.0, 0.0]))]
        if scenario not in ("stationary", "ego_motion"):
            states = [
                (
                    "object_1",
                    np.array([1 + 0.55 * t, -1 + 0.15 * t, 0.6]),
                    np.array([0.55, 0.15]),
                )
            ]
        if scenario == "crossing":
            states = [
                (
                    "object_1",
                    np.array([1 + 0.6 * t, -1 + 0.25 * t, 0.6]),
                    np.array([0.6, 0.25]),
                ),
                (
                    "object_2",
                    np.array([6 - 0.6 * t, 1 - 0.25 * t, 0.6]),
                    np.array([-0.6, -0.25]),
                ),
            ]
        detections = []
        truth = []
        for identifier, position, velocity in states:
            truth.append(
                dict(
                    object_id=identifier,
                    position=position.tolist(),
                    velocity=velocity.tolist(),
                    frame_id="odom",
                )
            )
            visible = not (scenario == "occlusion" and 30 <= index < 36) and not (
                scenario == "sparse" and index % 3 != 0
            )
            if visible:
                measured = transform_points(position[None], np.linalg.inv(pose))[
                    0
                ] + rng.normal(0, sigma, 3)
                detections.append(
                    dict(
                        detection_id=f"{index}:{len(detections)}",
                        position=measured.tolist(),
                        bbox_min=(measured - [0.25, 0.25, 0.5]).tolist(),
                        bbox_max=(measured + [0.25, 0.25, 0.5]).tolist(),
                        measurement_covariance=(np.eye(3) * sigma**2).tolist(),
                    )
                )
        rng.shuffle(
            detections
        )  # source ID/order are never used for cross-frame association
        rows.append(
            dict(
                timestamp=float(t),
                frame_id="base_link",
                clock_domain="synthetic",
                detections=detections,
                robot_pose=dict(
                    timestamp=float(t),
                    T_parent_child=pose.tolist(),
                    parent_frame="odom",
                    child_frame="base_link",
                    clock_domain="synthetic",
                ),
                ground_truth=truth,
            )
        )
    return jsonable(rows)
