"""Virtual-time kinematic proxy and image-based fixture. No model weights or robot I/O."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from copy import deepcopy
from hashlib import sha256
import math
from threading import Event
from time import monotonic, sleep

import numpy as np

from .continuous_planner import encode_view
from .continuous import ContinuousNavigator, PERIOD_S, simulated_velocity
from .episode_mock import effective_mock_capability
from .motion_profile import SENSOR, wrap


SCENARIOS = (
    "visible_after_search",
    "two_second_planner",
    "blocked_corridor",
    "lidar_dropout",
    "operator_cancel",
    "stopped_control_clock",
    "target_loss",
    "no_continuation",
)
CAMERA = {"width": 161, "fx": 225.0, "cx": 80.0}
START = 100.0


def sensor(pose, now, velocity=(0.0, 0.0), *, clearance=3.0, available=True):
    pose = dict(pose, monotonic_s=now)
    geom = dict(
        frame=pose["frame"],
        robot_epoch=pose["robot_epoch"],
        monotonic_s=now,
        available=available,
        source="synthetic observed room",
        directions=["forward", "left", "right"],
        forward_clear_m=clearance,
        turn_clear_rad=1.4,
    )
    return dict(
        available_s=now,
        snapshot=dict(
            pose=pose,
            geometry=geom,
            capture_clock_epoch="sim-camera-g0",
            capture_uncertainty_bound_s=0.0,
            capture_bound_basis="virtual common clock",
        ),
        interlock=dict(
            neutral=True, healthy=True, stationary=velocity == (0.0, 0.0), radio_s=now, state_s=now
        ),
    )


def render(pose):
    """One contrasting fixture chair in a pinhole field. Simulator truth stays here."""
    w, fx, cx = CAMERA["width"], CAMERA["fx"], CAMERA["cx"]
    image = np.zeros((101, w), dtype=np.uint8)
    dx, dy = 1.05 - pose["x_m"], 0.52 - pose["y_m"]
    c, s = math.cos(pose["yaw_rad"]), math.sin(pose["yaw_rad"])
    depth, left = c * dx + s * dy, c * dy - s * dx
    if depth > 0.12:
        x = round(cx - fx * left / depth)
        h = min(75, round(30 / depth))
        y, half = 50 - h // 2, max(2, h // 4)
        if half <= x < w - half:
            image[y : y + h // 2, x - half : x + half + 1] = 255
            image[y + h // 2 : y + 2 * h // 3, x - half : x + half + 1] = 255
            for leg in (x - half + 1, x + half - 2):
                image[y + 2 * h // 3 : y + h, leg : leg + 2] = 255
    return image


def fixture_result(template, guard, image, sequence, latency_s):
    """A slow worker stand-in observes pixels. No target pose, progress or goal signal is supplied."""
    ys, xs = np.nonzero(image)
    visible = len(xs) > 0
    height = int(ys.max() - ys.min() + 1) if visible else 0
    point = [float((xs.min() + xs.max() + 1) / 2 / image.shape[1]), 0.5] if visible else None
    stop = height >= 50
    extent = 0.60 if visible and not stop else 0.0
    yaw = 0.0 if visible else math.radians(30)
    snap = guard["snapshot"]
    p = deepcopy(template)
    t = guard["available_s"]
    pid = f"sim-P{sequence:04d}"
    p["identity"].update(
        prediction_id=pid,
        observation_id=f"sim-F{sequence:04d}",
        clock_epoch="sim-camera-g0",
        source_image_sha256=sha256(image.tobytes()).hexdigest(),
        observed_monotonic_s=t,
        received_monotonic_s=t,
        uncertainty_bound_s=0.0,
        pose=snap["pose"],
        geometry=snap["geometry"],
    )
    p.update(
        waypoints=[[extent * j / 10.0, 0.0, yaw * j / 10.0] for j in range(1, 11)],
        point_fraction=point,
        native_stop=stop,
        native_output=f"synthetic pixels visible={visible} height={height} stop={stop}",
    )
    evidence = dict(
        target="chair",
        image_sha256=p["identity"]["source_image_sha256"],
        prediction_id=pid,
        status="supported" if visible else "absent",
        method="synthetic_fixture",
        camera=CAMERA,
    )
    if visible:
        evidence["roi"] = [
            float(xs.min() / image.shape[1]),
            float(ys.min() / image.shape[0]),
            float((xs.max() + 1) / image.shape[1]),
            float((ys.max() + 1) / image.shape[0]),
        ]
    return dict(proposal=p, grounding=evidence, available_s=round(t + latency_s, 6))


def integrate(pose, packet, start_s, end_s):
    """Independent expiring sink integrates only the valid part of the preceding interval."""
    v, omega = simulated_velocity(packet, start_s)
    dt = max(0.0, min(end_s, packet["expires_s"]) - start_s) if packet else 0.0
    return dict(
        pose,
        x_m=pose["x_m"] + dt * v * math.cos(pose["yaw_rad"]),
        y_m=pose["y_m"] + dt * v * math.sin(pose["yaw_rad"]),
        yaw_rad=wrap(pose["yaw_rad"] + dt * omega),
        monotonic_s=end_s,
    )


def simulate(fixture, scenario, record, *, wall_clock=False, planner=None):
    """With wall_clock, one background visual worker can block while the calling thread checks guards."""
    close = Event()
    if planner is not None and not wall_clock:
        raise ValueError("actual LightNav requires wall-clock simulation")
    with ThreadPoolExecutor(max_workers=1) if wall_clock and planner is None else nullcontext() as pool:
        try:
            return _run(fixture, scenario, record, pool, close, planner)
        finally:
            close.set()


def _delayed_fixture(template, guard, image, sequence, delay, epoch, close):
    if close.wait(delay):
        return None
    result = fixture_result(template, guard, image, sequence, 0.0)
    result["available_s"] = START + (monotonic() - epoch)
    return result


def _run(fixture, scenario, record, pool, close, planner):
    if scenario not in SCENARIOS:
        raise ValueError("unknown scenario")
    cap = effective_mock_capability(deepcopy(fixture["capability"]))
    cap["turn_scope"] = SENSOR  # All three directions are modelled in this fixture.
    template = deepcopy(fixture["proposal"])
    template["identity"]["instruction"] = "Go to the chair and stop."
    pose = dict(template["identity"]["pose"], monotonic_s=START)
    scope = dict(template["identity"], clock_epoch="sim-camera-g0")
    nav = ContinuousNavigator(
        cap, scope, target="chair", operator="fixture", origin=pose, started_s=START, duration_s=12
    )
    delay = 2.0 if scenario == "two_second_planner" else 0.15
    pending = packet = None
    sequence = 0
    next_capture = START
    previous = START
    overlap = watchdog_intervals = 0
    proxy_travel = proxy_turn = 0.0
    first_forward = None
    end_reason = None
    epoch = next_wake = monotonic()
    control_times = []
    delivered_ages = []
    pending_id = None
    for step in range(242):
        if pool is None and planner is None:
            now = round(START + step * PERIOD_S, 6)
        else:
            sleep(max(0.0, next_wake - monotonic()))
            woke = monotonic()
            now = START + (woke - epoch)
            next_wake = next_wake + PERIOD_S
            if next_wake <= woke:
                next_wake = woke + PERIOD_S
        before = pose
        pose = integrate(pose, packet, previous, now)
        proxy_travel += math.hypot(pose["x_m"] - before["x_m"], pose["y_m"] - before["y_m"])
        proxy_turn += abs(wrap(pose["yaw_rad"] - before["yaw_rad"]))
        velocity = simulated_velocity(packet, now)
        if pending_id and (pose["x_m"], pose["y_m"], pose["yaw_rad"]) != (
            before["x_m"],
            before["y_m"],
            before["yaw_rad"],
        ):
            overlap += 1
        previous = now
        elapsed = None if first_forward is None else now - first_forward
        if scenario == "stopped_control_clock" and elapsed is not None and 0.04 <= elapsed < 0.36:
            if velocity == (0.0, 0.0):
                watchdog_intervals += 1
            record(dict(kind="missed_tick", t=now, pose=pose, sink_velocity=velocity))
            continue
        blocked = scenario == "blocked_corridor" and elapsed is not None and elapsed >= 0.09
        dropout = scenario == "lidar_dropout" and elapsed is not None and elapsed >= 0.09
        guard = sensor(pose, now, velocity, clearance=0.12 if blocked else 3.0, available=not dropout)
        delivery = None
        worker_error = None
        if planner is not None:
            try:
                delivery = planner.poll(now)
            except (RuntimeError, ValueError, OSError) as exc:
                worker_error = str(exc)
        elif pending is not None:
            ready = pending if pool is None else pending.result() if pending.done() else None
            if ready is not None and ready["available_s"] <= now:
                delivery, pending = ready, None
                delivery["fixture_delay_s"] = delay
        if delivery is not None:
            pending_id = None
            delivered_ages.append(now - delivery["proposal"]["identity"]["observed_monotonic_s"])
        if worker_error is None and now + 1e-8 >= next_capture:
            sequence += 1
            frame_id = f"sim-F{sequence:04d}"
            image = render(pose)
            if scenario == "target_loss" and elapsed is not None and elapsed >= 0.09:
                image.fill(0)
            png, view = encode_view(image)
            submit = pending_id is None and not (scenario == "no_continuation" and first_forward is not None)
            if planner is not None:
                try:
                    planner.capture(now, png, view["png_sha256"], frame_id)
                    if submit:
                        planner.submit(template, guard, frame_id, CAMERA, (START, epoch))
                except (RuntimeError, ValueError, OSError) as exc:
                    worker_error = str(exc)
                    submit = False
            elif submit:
                if pool is None:
                    pending = fixture_result(template, guard, image, sequence, delay)
                else:
                    pending = pool.submit(
                        _delayed_fixture, template, guard, image, sequence, delay, epoch, close
                    )
            if submit:
                pending_id = frame_id
            record(
                dict(kind="capture", t=now, observation_id=frame_id, pose=pose, view=view, submitted=submit)
            )
            next_capture += 0.25
            if next_capture <= now:
                next_capture = now + 0.25
        control_times.append(now)
        cancel = scenario == "operator_cancel" and elapsed is not None and elapsed >= 0.09
        packet = nav.tick(
            now, guard, delivery=delivery, cancel=cancel, worker_failed=worker_error is not None
        )
        if first_forward is None and packet["velocity"][0] > 0:
            first_forward = now
        record(
            dict(
                kind="tick",
                t=now,
                guard=guard,
                delivery=delivery,
                packet=packet,
                pose=pose,
                pending_observation=pending_id,
                worker_error=worker_error,
            )
        )
        if nav.terminal:
            end_reason = nav.terminal
            break
    return dict(
        source="lightnav_on_simulated_camera" if planner else "synthetic_image_fixture",
        distinct_camera_observations=sequence,
        scenario=scenario,
        end_reason=end_reason,
        worker_latency_s=None if planner else delay,
        learned_jobs=dict(
            submitted=planner.submitted,
            completed=planner.completed,
            max_resident_camera_images=planner.max_images,
        )
        if planner
        else None,
        period_s=PERIOD_S,
        clock="host_monotonic" if pool or planner else "virtual",
        control_ticks=len(control_times),
        control_gap_p95_s=float(np.percentile(np.diff(control_times), 95))
        if len(control_times) > 1
        else None,
        control_gap_max_s=max(np.diff(control_times), default=None),
        result_age_at_poll_s=delivered_ages,
        started_s=START,
        ended_s=now,
        plans=nav.plans,
        uninterrupted_replacements=nav.rollovers,
        moving_intervals_while_inference_pending=overlap,
        watchdog_zero_intervals=watchdog_intervals,
        odom_travel_m=nav.travel,
        odom_turn_rad=nav.turn,
        proxy_path_length_m=proxy_travel,
        proxy_absolute_yaw_rad=proxy_turn,
        proxy_final_pose=pose,
        physical_commands_sent=0,
    )
