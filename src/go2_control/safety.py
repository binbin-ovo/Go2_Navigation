"""Pure safety logic for one bounded high-level move. No SDK import, no I/O beyond the injected
route, monitor, clock, sleep and log callables, so every path is testable with mocks.

The robot-side lifetime of a Move is documented for the sport service only ("the latest Move
command will be maintained for 1 second", Unitree Motion Services Interface V2.0, firmware >= 1.1.6).
The obstacles_avoid route lifetime is undocumented and must be measured; see
docs/GO2_HIGH_LEVEL_CONTROL.md. This module does not invent a watchdog: the only timing rules
are the caller's finite duration (monotonic deadline) and an optional resend period that must stay
well below the documented 1 s hold.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

# Conservative first-test limits. Unitree's documented ranges are far wider (vx -2.5..3.8 m/s,
# vyaw -4..4 rad/s); these caps are ours.
# 0.30 m/s walked (C4) but a 0.14 m step ended inside the ~1.3 s ramp and the 15 deg turns at 1.0 rad/s
# were not visibly distinguishable to the operator (2026-11-18). 1.0 m/s and 2.0 rad/s are the
# user-requested next step; above the previous caps only with a measured stop (distance/heading target).
MAX_VX = 1.0  # m/s, forward
MAX_TIMED_VX = 0.30  # m/s; faster forward only with a distance target
# Reverse: slower, and only with a measured distance stop. cloud_base had zero returns in a
# 0.7 m-wide corridor behind the robot out to 4 m (2026-11-15), so nothing on board sees there.
# 0.15 m/s forward never visibly walked (C, C2) while 0.30 did (C4), so a gait threshold is
# plausible. Reverse at 0.25 stalled from R2 on; 0.35 started promptly but held after 0.19 m (X3c).
# 0.50 backed 0.64 m cleanly once (X3d) then held after 0.12 m (R3). 0.75 is the user-requested
# next step, still only with a distance stop, stall stop and short cap. The route's realized speed
# has been ~70 % of the command.
MAX_REVERSE_VX = 0.75  # m/s
# D1/D2 (2026-11-15) at 0.50 rad/s reached only 0.24-0.62 rad/s and right turns ran slower, the same
# under-delivery that made slow reverse stall. 1.0 is the user-requested step, with a heading stop.
MAX_ABS_VYAW = 2.0  # rad/s
MAX_TIMED_VYAW = 0.50  # rad/s; faster turns only with a heading target
MAX_YAW_TARGET = 1.6  # rad (~90 deg)
MAX_DURATION = 2.0  # s
MIN_DURATION = 0.2  # s
MAX_RESEND_PERIOD = 0.5  # s, half the documented 1 s hold
MIN_RESEND_PERIOD = 0.05  # s
STATE_MAX_AGE = 0.25  # s; rt/sportmodestate arrives at about 295 Hz on this robot
STICK_DEADBAND = 0.10  # normalized stick units
# Distance-goal moves: forward only, stopped by measured progress, still bounded by a deadline.
# On 2026-11-15 a 2.0 s, 0.30 m/s command took ~1.8 s to reach speed and covered 0.37 m (tape).
MAX_TARGET = 1.6  # m
MAX_TARGET_DURATION = 8.0  # s
# Stall: in R2 (2026-11-15) the robot stopped after 0.27 m while reverse commands continued for
# 5 s. Measured progress then wandered by <= 0.04 m (LiDAR noise), so 0.06 m over 1.5 s is a stall.
STALL_GRACE = 2.0  # s after the first command; the robot takes ~0.4 s to start and ramps slowly
MAX_STALL_GRACE = 6.0  # s; a longer grace only for testing whether reverse starts late
STALL_WINDOW = 1.5  # s
STALL_MIN_PROGRESS = 0.06  # m


class Refused(Exception):
    """A precondition failed; nothing was sent."""


@dataclass(frozen=True)
class MoveRequest:
    vx: float  # body-frame forward speed, m/s
    vyaw: float  # body yaw rate, rad/s; positive = counter-clockwise (left), verified D1/D2 2026-11-15
    duration: float  # s, finite
    resend_period: float = 0.0  # 0 = send once, then stop at the deadline
    target: float = 0.0  # progress in the commanded direction (m, or rad for a turn); 0 = none

    def describe(self) -> str:
        cadence = "once" if self.resend_period == 0 else f"every {self.resend_period:.2f}s"
        text = f"vx={self.vx:.2f} vy=0.00 vyaw={self.vyaw:.2f} for {self.duration:.1f}s, sent {cadence}"
        unit = "m" if self.vx else "rad"
        return text + (f", stop at {self.target:.2f}{unit}" if self.target else "")


def validate(req: MoveRequest) -> MoveRequest:
    """Reject anything outside the first-test envelope. Lateral motion is not representable."""
    values = (req.vx, req.vyaw, req.duration, req.resend_period, req.target)
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
        raise Refused("all values must be finite numbers")
    if req.vx < -MAX_REVERSE_VX:
        raise Refused(f"reverse speed {-req.vx} exceeds {MAX_REVERSE_VX} m/s")
    if req.vx < 0 and not req.target:
        raise Refused("reverse motion needs a measured distance target")
    if req.vx > MAX_VX:
        raise Refused(f"vx {req.vx} exceeds {MAX_VX} m/s")
    if req.vx > MAX_TIMED_VX and not req.target:
        raise Refused(f"forward faster than {MAX_TIMED_VX} m/s needs a distance target")
    if abs(req.vyaw) > MAX_ABS_VYAW:
        raise Refused(f"|vyaw| {abs(req.vyaw)} exceeds {MAX_ABS_VYAW} rad/s")
    if abs(req.vyaw) > MAX_TIMED_VYAW and not req.target:
        raise Refused(f"turns faster than {MAX_TIMED_VYAW} rad/s need a heading target")
    if req.vx != 0 and req.vyaw != 0:
        raise Refused("combined translation and turn is not allowed")
    if req.vx == 0 and req.vyaw == 0:
        raise Refused("a zero request is not a move; use the stop mode")
    if req.target:
        limit, unit = (MAX_TARGET, "m") if req.vx else (MAX_YAW_TARGET, "rad")
        if not 0 < req.target <= limit:
            raise Refused(f"target must be in (0, {limit}] {unit}")
        if req.resend_period == 0:
            raise Refused("a distance target needs a resend period")
        max_duration = MAX_TARGET_DURATION
    else:
        max_duration = MAX_DURATION
    if not MIN_DURATION <= req.duration <= max_duration:
        raise Refused(f"duration must be in [{MIN_DURATION}, {max_duration}] s")
    if req.resend_period != 0 and not MIN_RESEND_PERIOD <= req.resend_period <= MAX_RESEND_PERIOD:
        raise Refused(f"resend period must be 0 or in [{MIN_RESEND_PERIOD}, {MAX_RESEND_PERIOD}] s")
    return req


def confirmation_phrase(req: MoveRequest) -> str:
    """Exact text the operator must pass back, so the armed command matches the approved test."""
    return f"MOVE {req.describe()}"


class Route(Protocol):
    """Only what a bounded move needs. Deliberately has no avoidance switch-set method."""

    def avoidance_enabled(self) -> Optional[bool]: ...  # fresh SwitchGet; None = unknown
    def claim_api_control(self) -> int: ...  # UseRemoteCommandFromApi(True) -> code
    def release_api_control(self) -> int: ...  # UseRemoteCommandFromApi(False) -> code
    def send_velocity(self, vx: float, vyaw: float) -> int: ...  # non-blocking, no reply
    def send_stop(self) -> int: ...  # the verified stop for this route, non-blocking


class Monitor(Protocol):
    def state_age(self) -> Optional[float]: ...  # s since last sport state; None = never
    def operator_input(self) -> bool: ...  # sticks or buttons active on the remote
    def is_still(self) -> Optional[bool]: ...  # body speed and yaw rate near zero; None = unknown


@dataclass
class Outcome:
    started: bool = False
    progress_last: Optional[float] = None
    nonzero_sent: int = 0
    stop_sent: int = 0
    end_reason: str = ""
    stop_confirmed: Optional[bool] = None
    released: Optional[int] = None
    events: list = field(default_factory=list)


def preflight(route: Route, monitor: Monitor, *, verified_stop: bool, other_source: bool) -> None:
    if not verified_stop:
        raise Refused("the stop for this route is not hardware-verified")
    if other_source:
        raise Refused("another software command source is active")
    enabled = route.avoidance_enabled()
    if enabled is None:
        raise Refused("avoidance state unknown (query failed)")
    if enabled is not True:
        raise Refused("avoidance is disabled; enabling needs separate user approval")
    age = monitor.state_age()
    if age is None or age > STATE_MAX_AGE:
        raise Refused(f"robot state is stale or missing (age={age})")
    if monitor.operator_input():
        raise Refused("operator is using the remote")


def execute(
    req: MoveRequest,
    route: Route,
    monitor: Monitor,
    *,
    confirm: str,
    execute_flag: bool,
    verified_stop: bool,
    other_source: bool = False,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[dict], None] = lambda e: None,
    progress: Optional[Callable[[], Optional[float]]] = None,
    stop_repeats: int = 3,
    stop_spacing: float = 0.05,
    stall_grace: float = STALL_GRACE,
    settle_timeout: float = 1.5,
) -> Outcome:
    """One bounded move. Never resumes: after any stop, timeout or error this returns."""
    validate(req)
    if req.target and not STALL_GRACE <= stall_grace <= min(MAX_STALL_GRACE, req.duration):
        raise Refused(f"stall grace must be in [{STALL_GRACE}, {MAX_STALL_GRACE}] s and within the duration")
    if not execute_flag:
        raise Refused("dry run: --execute not given")
    if confirm != confirmation_phrase(req):
        raise Refused("confirmation text does not match this exact request")
    if req.target and progress is None:
        raise Refused("a distance target needs a progress measurement")
    preflight(route, monitor, verified_stop=verified_stop, other_source=other_source)
    if req.target and progress() is None:
        raise Refused("progress measurement unavailable before the move")

    out = Outcome()

    def note(kind, **kw):
        event = {"t": clock(), "event": kind, **kw}
        out.events.append(event)
        log(event)

    code = route.claim_api_control()
    note("claim_api_control", code=code)
    if code != 0:
        out.end_reason = f"claim failed ({code})"
        out.released = route.release_api_control()
        note("release_api_control", code=out.released)
        return out

    out.started = True
    start = clock()
    deadline = start + req.duration
    history = []  # (t, progress) for the stall check
    stopped = False
    try:
        out.end_reason = "deadline"
        while True:
            now = clock()
            if now >= deadline:
                break
            if monitor.operator_input():
                out.end_reason = "operator input"
                break
            age = monitor.state_age()
            if age is None or age > STATE_MAX_AGE:
                out.end_reason = f"stale state ({age})"
                break
            if req.target:
                done = progress()
                if done is None:
                    out.end_reason = "progress unavailable"
                    break
                out.progress_last = done
                if done >= req.target:
                    out.end_reason = f"target reached ({done:.3f} {'m' if req.vx else 'rad'})"
                    break
                history.append((now, done))
                old = [p for t, p in history if t <= now - STALL_WINDOW]
                if now - start >= stall_grace and old and done - old[-1] < STALL_MIN_PROGRESS:
                    out.end_reason = f"stalled at {done:.3f} m (robot not progressing)"
                    break
            if out.nonzero_sent == 0 or req.resend_period > 0:
                note("send_velocity", vx=req.vx, vyaw=req.vyaw, code=route.send_velocity(req.vx, req.vyaw))
                out.nonzero_sent += 1
            step = req.resend_period if req.resend_period > 0 else 0.02
            sleep(max(0.0, min(step, deadline - clock())))
    except BaseException as exc:  # includes KeyboardInterrupt and cancellation
        out.end_reason = f"exception: {type(exc).__name__}: {exc}"
        _stop(route, out, note, stop_repeats, 0.0, sleep)
        stopped = True
        raise
    finally:
        if not stopped:
            _stop(route, out, note, stop_repeats, stop_spacing, sleep)
        out.stop_confirmed = _await_still(monitor, clock, sleep, settle_timeout)
        note("stop_observed", confirmed=out.stop_confirmed)
        out.released = route.release_api_control()
        note("release_api_control", code=out.released)
    return out


def _stop(route, out, note, repeats, spacing, sleep):
    for i in range(repeats):
        try:
            code = route.send_stop()
        except Exception as exc:  # keep trying; never block here
            code = f"error: {exc}"
        out.stop_sent += 1
        note("send_stop", code=code)
        if spacing and i + 1 < repeats:
            sleep(spacing)


def _await_still(monitor, clock, sleep, timeout) -> Optional[bool]:
    end = clock() + timeout
    while clock() < end:
        still = monitor.is_still()
        if still:
            return True
        sleep(0.02)
    return monitor.is_still()
