"""Opt-in stable scene pipeline. Every new scan is processed before rendering decisions."""

import time
import numpy as np
from .accumulation import TemporalAccumulator
from .background import BackgroundMap
from .objects import SceneObjects
from .proposals import make_proposals
from .serialization import validate_scene
from go2_perception.lidar.ground_segmentation import ground_mask


class StableScene:
    def __init__(self, config):
        self.config = config
        self.accumulator = TemporalAccumulator(**config["accumulation"])
        self.background = BackgroundMap(**config["background"])
        self.objects = SceneObjects(**config["objects"])

    def process(
        self,
        xyz,
        stamp,
        scan_id,
        pose_lookup,
        *,
        clock_domain,
        point_offsets=None,
        deskew=False,
        instantaneous_point_frame=False,
        reset=False,
    ):
        start = time.perf_counter()
        c = self.config
        a = self.accumulator
        new = a.add(
            xyz,
            stamp,
            scan_id,
            pose_lookup,
            clock_domain=clock_domain,
            point_offsets=point_offsets,
            deskew=deskew,
            instantaneous_point_frame=instantaneous_point_frame,
            reset=reset,
        )
        if a.reset_reason:
            self.background.clear()
            self.objects.clear()
        now = (
            float(stamp + np.max(point_offsets))
            if point_offsets is not None and len(point_offsets)
            else float(stamp)
        )
        diagnostics = dict(
            synthetic=clock_domain == "synthetic",
            reset_reason=a.reset_reason,
            timing_mode="point_pose"
            if deskew
            else "message_midpoint_pose"
            if point_offsets is not None
            else "message_stamp_pose",
            measurement_policy="all_new_points_processed",
            raw_display_suppression="renderer_only",
            point_timestamps_available=point_offsets is not None,
            background_min_observations=self.background.min_observations,
            background_ttl_s=self.background.ttl,
            object_min_observations=3,
            object_ttl_s=self.objects.settings["ttl_s"],
        )
        if new is not None:
            now = new.timestamp
            pose = a.last_pose

            def nonground(p):
                p = p[np.linalg.norm(p - pose[:3, 3], axis=1) <= c["max_range_m"]]
                mask = ground_mask(
                    p,
                    expected_ground_z=pose[2, 3] - c["sensor_height_m"],
                    **c["ground"],
                )
                return p[~mask]

            window = a.snapshot(c["object_window_s"], c["voxel_size"])
            p = nonground(window.xyz)
            proposals, details = make_proposals(
                p, background=self.background, timestamp=now, **c["proposals"]
            )
            current = nonground(new.xyz)
            supported = []
            for proposal in proposals:
                support = current[
                    (
                        (current >= proposal["low"] - 0.05)
                        & (current <= proposal["high"] + 0.05)
                    ).all(1)
                ]
                if len(support) < c.get("current_support_min", 3):
                    continue
                low, high = np.percentile(support, [2, 98], axis=0)
                proposal["observed_box"] = dict(
                    center=((low + high) / 2).tolist(),
                    dimensions=np.maximum(high - low, 0.03).tolist(),
                    point_count=len(support),
                )
                supported.append(proposal)
            details["unsupported_old_proposals"] = len(proposals) - len(supported)
            details["proposal_boxes"] = [
                dict(center=p["center"].tolist(), dimensions=p["dimensions"].tolist())
                for p in supported
            ]
            proposals = supported
            self.objects.update(proposals, now)
            objects = self.objects.snapshot(now)
            moving = []
            for o in objects:
                if o["motion_state"] == "MOVING":
                    center = np.asarray(o["center"])
                    dim = np.asarray(o["dimensions"])
                    moving.append((center - dim / 2, center + dim / 2))
            self.background.update(current, now, scan_id, pose[:3, 3], moving)
            diagnostics.update(
                details,
                accumulated_points=len(window.xyz),
                acquisition_age_max_s=float(window.ages.max())
                if len(window.ages)
                else 0.0,
                source_intervals=len(set(window.scan_ids)),
                new_points=len(current),
            )
            robot_pose = dict(
                timestamp=now,
                parent_frame="odom",
                child_frame="base_link",
                T_odom_base=pose.tolist(),
            )
        else:
            objects = []
            robot_pose = None
        diagnostics["processing_ms"] = (time.perf_counter() - start) * 1000
        row = dict(
            schema="go2-scene-v1",
            timestamp=now,
            clock_domain=clock_domain,
            frame_id="odom",
            units="m",
            epoch=a.epoch,
            robot_pose=robot_pose,
            background=self.background.snapshot(now),
            objects=objects,
            diagnostics=diagnostics,
        )
        return validate_scene(row)
