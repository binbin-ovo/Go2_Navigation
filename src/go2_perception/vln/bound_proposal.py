"""Bind a native proposal to its archived image, capture-time guard snapshot and model."""

import hashlib
import json
from queue import Queue

from .supervised import Hold, MODEL_REVISION


class BoundProposalUI:
    def __init__(self, app, cap, live, executor):
        self.app, self.cap, self.live, self.executor = app, cap, live, executor
        self.queue = Queue(maxsize=16)
        self.instruction_id = None

    def _identity(self, pred):
        frame, image = pred["source_frame"], pred["source_image"]
        observed = frame.get("exposure_monotonic_s")
        if not self.live:
            anchor = self.app.session.replay_anchor
            if anchor is None:
                raise Hold("synthetic_clock_mapping_unavailable")
            observed = anchor[1] + (frame["observation_s"] - anchor[0]) / anchor[2]
        if observed is None:
            raise Hold("capture_monotonic_mapping_unavailable")
        guard = self.executor.guard_at(observed, self.cap["limits"]["state_age_s"])
        if self.live and (
            guard.get("capture_clock_epoch") != frame["clock_epoch"]
            or not guard.get("capture_bound_basis")
        ):
            raise Hold("capture_age_bound_not_validated_for_this_camera_epoch")
        return {
            "session_id": self.app.session.session_id,
            "trial_id": pred["trial_id"],
            "instruction_id": self.instruction_id,
            "instruction": pred["instruction"],
            "prediction_id": pred["prediction_id"],
            "observation_id": frame["frame_id"],
            "clock_epoch": frame["clock_epoch"],
            "source_image_sha256": image["file_sha256"],
            "observed_monotonic_s": observed,
            "received_monotonic_s": frame["received_monotonic_s"],
            "uncertainty_bound_s": guard.get("capture_uncertainty_bound_s"),
            "pose": guard["pose"],
            "geometry": guard["geometry"],
        }

    def proposal(self, pred):
        if not pred.get("source_image"):
            raise Hold("archived_source_image_unavailable")
        path = self.app.store.root / pred["source_image"]["path"]
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != pred["source_image"]["file_sha256"]
        ):
            raise Hold("source_image_changed")
        if (
            pred["waypoint_convention"]
            != "absolute_ego_SE2_x_forward_y_left_yaw_CCW_metres_radians"
        ):
            raise Hold("unsupported_waypoint_convention")
        if pred["rejection_reason_at_receipt"]:
            raise Hold("model_result_rejected: " + pred["rejection_reason_at_receipt"])
        if self.live:
            identity_path = self.app.store.root / "session/worker/model.json"
            identity = json.loads(identity_path.read_text())
            # Keep source identity in archives; require the pinned checkpoint in its identity.
            if (
                identity.get("checkpoint") != MODEL_REVISION
                or identity.get("upstream_code")
                != "3015508b70fb6e30bbd66701e94c9b916dcb8df8"
                or not identity.get("processor")
            ):
                raise Hold("pinned_model_identity_not_verified")
        revision = MODEL_REVISION if self.live else "synthetic"
        proposal = {
            "identity": self._identity(pred),
            "native_output": pred["native_output"],
            "model_identity": {
                "checkpoint": "LightOriginsHQ/LightNav-0"
                if self.live
                else "synthetic-mock",
                "revision": revision,
                "processor_revision": revision,
            },
            "waypoints": pred["waypoints"],
            "native_stop": pred["native_stop"],
            "point_fraction": None
            if pred["target_fraction"] is None
            else list(pred["target_fraction"]),
            "synthetic": pred["mock"],
        }
        return proposal, path
