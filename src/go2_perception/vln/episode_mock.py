"""Explicitly synthetic decisions for exercising the same episode IPC and recorder."""

import math

import cv2

from .shadow_display import point_fraction
from .shadow_sources import SyntheticSource

from .motion_profile import INSPECTED, LIMITS, PROFILE
from .shadow import write_json
from .shadow_worker import MockWorker


def effective_mock_capability(cap):
    cap.update(motion_profile=PROFILE, turn_scope=INSPECTED)
    cap["limits"].update(**LIMITS, stationary_dwell_s=1.0, lateral_m=0.08, straight_yaw_rad=0.10)
    cap["mapping"].update(
        verified_actions=["forward"],
        translation_m=None,
        translation_bound_m=0.5,
        yaw_rad=0.0,
        yaw_uncertainty_rad=0.05,
        orientation_basis="relative_planar_heading_ccw",
    )
    return cap


class EpisodeMockScene:
    """A canned view changes only after the expected fake action produces settled progress."""

    def __init__(self, fixture):
        self.fixture = fixture
        self.view = 0

    def settled(self, direction, result):
        m = result.get("motion_measurement") or {}
        case = self.fixture["decisions"][self.view]
        if (
            case.get("advance_after") == direction
            and m.get("stop_verified")
            and m.get("outcome") in ("partial_progress", "target_within_tolerance", "overshoot")
        ):
            self.view = min(self.view + 1, len(self.fixture["decisions"]) - 1)

    def source(self, frames, fps):
        for delivery in SyntheticSource(frames, fps):
            view = self.view
            image = delivery.observation.visual.image
            image[:160] = (220, 224, 225)
            image[160:] = (155, 150, 145)
            cv2.rectangle(image, (335, 46), (408, 159), (144, 111, 88), 4)
            point = point_fraction(self.fixture["decisions"][view]["opos"])
            if point:
                x, y = round(point[0] * 448), round(point[1] * 256)
                scale = 1.25 if view == len(self.fixture["decisions"]) - 1 else 1.0

                def xy(dx, dy):
                    return (int(x + dx * scale), int(y + dy * scale))

                color = (99, 57, 34)
                cv2.rectangle(image, xy(-21, -22), xy(21, 9), color, -1)
                cv2.rectangle(image, xy(-24, 15), xy(24, 22), color, -1)
                for dx in (-19, 19):
                    cv2.line(image, xy(dx, 6), xy(dx, 43), color, 4)
            delivery.provenance["synthetic_fixture_view"] = view
            yield delivery


class EpisodeMockWorker(MockWorker):
    def __init__(self, fixture):
        super().__init__(0.15)
        self.decisions = fixture["decisions"]

    def poll(self, now):
        request = self.pending[0] if self.pending else None
        response = super().poll(now)
        if response is None:
            return None
        native = response["result"]
        index = request.context["current_source"]["provenance"]["synthetic_fixture_view"]
        case = self.decisions[index]
        native["pointing"]["opos"] = case["opos"]
        native["native_output"] = native["native_output"].replace("opos_0", f"opos_{case['opos']}")
        native["waypoints"] = [
            [case["extent_m"] * i / 10.0, 0.0, math.radians(case["yaw_deg"]) * i / 10.0] for i in range(1, 11)
        ]
        # The decoder may return a zero chunk without an explicit special token.
        native["decoder_stop"] = case["stop"]
        write_json(request.folder / "mock_result.json", native)
        return response
