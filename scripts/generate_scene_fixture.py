"""Standalone deterministic renderer fixture; no sensor/perception imports needed."""

import argparse
from copy import deepcopy
from go2_perception.scene.serialization import write_scenes


def fixture():
    for i in range(6):
        t = i * 0.2
        pose = {
            "timestamp": t,
            "parent_frame": "odom",
            "child_frame": "base_link",
            "T_odom_base": [
                [1, 0, 0, 0.1 * t],
                [0, 1, 0, 0],
                [0, 0, 1, 0.3],
                [0, 0, 0, 1],
            ],
        }
        objects = []
        for ident, center, velocity in [
            (1, [2, 0, 0.4], [0, 0, 0]),
            (2, [1.5, t - 0.5, 0.8], [0, 1, 0]),
        ]:
            stale = ident == 1 and i >= 4
            last = 0.6 if stale else t
            obj = dict(
                object_id=ident,
                track_id=None,
                center=center,
                dimensions=[0.5, 0.5, 0.8 if ident == 1 else 1.6],
                orientation_xyzw=None,
                velocity=velocity,
                velocity_covariance=[[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]],
                motion_state="UNCERTAIN"
                if stale or i < 2
                else "STATIONARY"
                if ident == 1
                else "MOVING",
                status="stale" if stale else "observed_filtered",
                last_observed=last,
                observation_age_s=t - last,
                age_s=t,
                observation_score=min((i + 1) / 3, 1.0)
                * max(0.0, 1 - (t - last) / 1.0),
                observed_box=None
                if stale
                else {
                    "center": deepcopy(center),
                    "dimensions": [0.5, 0.5, 0.8 if ident == 1 else 1.6],
                    "point_count": 20,
                },
                display_compact=i >= 2 and not stale,
            )
            objects.append(obj)
        yield dict(
            schema="go2-scene-v1",
            timestamp=t,
            clock_domain="synthetic",
            frame_id="odom",
            units="m",
            epoch=0,
            robot_pose=pose,
            background={
                "voxel_size": 0.1,
                "free_space_policy": "unobserved_is_unknown",
                "cells": [
                    {
                        "center": [3.0, float(y), 0.5],
                        "observation_count": i + 1,
                        "last_observed": t,
                        "age_s": 0.0,
                        "evidence_score": min((i + 1) / 3, 1.0),
                    }
                    for y in [-1, 0, 1]
                ],
            },
            objects=objects,
            diagnostics={
                "synthetic": True,
                "min_observations": 3,
                "ttl_s": 1.0,
                "reset_reason": None,
                "timing_mode": "message_pose",
                "measurement_policy": "all_new_points_processed",
                "raw_display_suppression": "renderer_only",
                "processing_ms": 0.0,
            },
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    write_scenes(a.output, fixture())
