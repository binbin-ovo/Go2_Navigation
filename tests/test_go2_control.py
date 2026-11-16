"""Mock tests for the guarded Go2 command utility. No robot, no DDS."""
import subprocess
import sys
from pathlib import Path

import pytest

from go2_control.safety import MoveRequest, Refused, confirmation_phrase, execute, validate

ROOT = Path(__file__).resolve().parents[1]


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def sleep(self, dt):
        self.t += max(dt, 0.001)


class FakeRoute:
    def __init__(self, enabled=True, claim=0, fail_on_send=None):
        self.enabled = enabled
        self.claim = claim
        self.calls = []
        self.fail_on_send = fail_on_send

    def avoidance_enabled(self):
        self.calls.append("switch_get")
        return self.enabled

    def claim_api_control(self):
        self.calls.append("claim")
        return self.claim

    def release_api_control(self):
        self.calls.append("release")
        return 0

    def send_velocity(self, vx, vyaw):
        self.calls.append(("vel", vx, vyaw))
        if self.fail_on_send and len([c for c in self.calls if c[0] == "vel"]) == self.fail_on_send:
            raise KeyboardInterrupt
        return 0

    def send_stop(self):
        self.calls.append("stop")
        return 0


class FakeMonitor:
    def __init__(self, age=0.01, operator=False, still=True, operator_after=None, clock=None):
        self.age, self.operator, self.still = age, operator, still
        self.operator_after, self.clock = operator_after, clock

    def state_age(self):
        return self.age

    def operator_input(self):
        if self.operator_after is not None and self.clock() >= self.operator_after:
            return True
        return self.operator

    def is_still(self):
        return self.still


FWD = MoveRequest(vx=0.15, vyaw=0.0, duration=1.0)


def run(req=FWD, route=None, monitor=None, clock=None, **kw):
    clock = clock or FakeClock()
    route = route or FakeRoute()
    monitor = monitor or FakeMonitor()
    args = dict(confirm=confirmation_phrase(req), execute_flag=True, verified_stop=True)
    args.update(kw)
    return execute(req, route, monitor, clock=clock, sleep=clock.sleep, **args), route


def test_dry_run_sends_nothing():
    route = FakeRoute()
    with pytest.raises(Refused, match="dry run"):
        run(route=route, execute_flag=False)
    assert route.calls == []


def test_preview_cli_opens_no_dds_or_sdk():
    code = (
        "import sys, runpy; sys.argv=['go2_hl_control.py','preview','--vx','0.15','--duration','1'];"
        "import builtins; m=runpy.run_path('scripts/go2_hl_control.py', run_name='not_main');"
        "rc=m['main'](sys.argv[1:]); assert rc==0;"
        "assert 'unitree_sdk2py' not in sys.modules and 'cyclonedds' not in sys.modules"
    )
    env = {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"}
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT, env=env)


@pytest.mark.parametrize("enabled", [None, False])
def test_unknown_or_disabled_avoidance_blocks(enabled):
    route = FakeRoute(enabled=enabled)
    with pytest.raises(Refused, match="avoidance"):
        run(route=route)
    assert route.calls == ["switch_get"]


@pytest.mark.parametrize(
    "req",
    [
        MoveRequest(-0.1, 0.0, 1.0),  # reverse
        MoveRequest(0.5, 0.0, 1.0),  # too fast
        MoveRequest(0.0, 1.0, 1.0),  # turn too fast
        MoveRequest(0.1, 0.2, 1.0),  # combined
        MoveRequest(0.0, 0.0, 1.0),  # zero is not a move
        MoveRequest(0.1, 0.0, 0.0),  # zero duration
        MoveRequest(0.1, 0.0, 5.0),  # too long
        MoveRequest(0.1, 0.0, float("inf")),
        MoveRequest(float("nan"), 0.0, 1.0),
        MoveRequest(0.1, 0.0, 1.0, resend_period=1.0),  # longer than half the 1 s hold
    ],
)
def test_invalid_limits_rejected(req):
    with pytest.raises(Refused):
        validate(req)


def test_confirmation_must_match_exact_request():
    route = FakeRoute()
    other = confirmation_phrase(MoveRequest(0.2, 0.0, 1.0))
    with pytest.raises(Refused, match="confirmation"):
        run(route=route, confirm=other)
    assert route.calls == []


def test_unverified_stop_stale_state_operator_or_other_source_block():
    for kw, mon in [
        ({"verified_stop": False}, FakeMonitor()),
        ({"other_source": True}, FakeMonitor()),
        ({}, FakeMonitor(age=None)),
        ({}, FakeMonitor(age=1.0)),
        ({}, FakeMonitor(operator=True)),
    ]:
        route = FakeRoute()
        with pytest.raises(Refused):
            run(route=route, monitor=mon, **kw)
        assert not any(isinstance(c, tuple) for c in route.calls)


def test_send_once_then_stop_and_release_in_order():
    out, route = run()
    vel = [c for c in route.calls if isinstance(c, tuple)]
    assert vel == [("vel", 0.15, 0.0)]
    tail = route.calls[route.calls.index(vel[-1]) + 1:]
    assert tail == ["stop", "stop", "stop", "release"]
    assert out.end_reason == "deadline" and out.stop_confirmed is True


def test_resend_cadence_bounded_by_deadline():
    req = MoveRequest(0.15, 0.0, 1.0, resend_period=0.2)
    out, route = run(req=req)
    assert out.nonzero_sent == 5


def test_operator_input_aborts_and_stops():
    clock = FakeClock()
    mon = FakeMonitor(operator_after=clock.t + 0.3, clock=clock)
    out, route = run(clock=clock, monitor=mon)
    assert out.end_reason == "operator input"
    assert route.calls[-4:] == ["stop", "stop", "stop", "release"]


def test_exception_attempts_stop_and_release_then_reraises():
    req = MoveRequest(0.15, 0.0, 1.0, resend_period=0.2)
    route = FakeRoute(fail_on_send=2)
    clock = FakeClock()
    with pytest.raises(KeyboardInterrupt):
        execute(req, route, FakeMonitor(), confirm=confirmation_phrase(req), execute_flag=True,
                verified_stop=True, clock=clock, sleep=clock.sleep)
    assert route.calls[-4:] == ["stop", "stop", "stop", "release"]


def test_no_nonzero_after_stop_even_if_state_returns():
    """A stale-state abort (e.g. link drop) must not resume when state comes back."""
    clock = FakeClock()

    class Flaky(FakeMonitor):
        def state_age(self):
            return 5.0 if 100.3 <= clock.t < 100.5 else 0.01

    req = MoveRequest(0.15, 0.0, 1.0, resend_period=0.1)
    out, route = run(req=req, clock=clock, monitor=Flaky())
    first_stop = route.calls.index("stop")
    assert not any(isinstance(c, tuple) for c in route.calls[first_stop:])
    assert out.end_reason.startswith("stale state")


def test_unconfirmed_stop_is_reported():
    out, _ = run(monitor=FakeMonitor(still=False))
    assert out.stop_confirmed is False


def test_failed_claim_sends_no_velocity():
    out, route = run(route=FakeRoute(claim=3104))
    assert not out.started and route.calls == ["switch_get", "claim", "release"]


def test_no_disable_call_is_possible():
    src = (ROOT / "src/go2_control/sdk_routes.py").read_text()
    script = (ROOT / "scripts/go2_hl_control.py").read_text()
    code_lines = [l for l in src.splitlines() if not l.strip().startswith("#")]
    registered = "\n".join(l for l in code_lines if "[" in l and "AVOID_" in l)
    assert "AVOID_SWITCH_SET" not in registered
    for full in (src, script):
        # import and call lines only, as in test_go2_adapter's publisher guard; docstrings may name them
        text = "\n".join(l for l in full.splitlines() if "import" in l or ("(" in l and not l.strip().startswith(("#", "\"", "'"))))
        assert "SwitchSet(" not in text and "SwitchAvoidMode" not in text and "FreeAvoid" not in text
        assert "SportClient" not in text and "Damp" not in text and "StandDown" not in text
    assert '{"enable"' not in src and "'enable':" not in src  # no SwitchSet payload is built


def test_package_import_is_inert_and_perception_never_imports_control():
    code = (
        "import sys, go2_control, go2_control.safety; "
        "assert 'unitree_sdk2py' not in sys.modules and 'cyclonedds' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, env={"PYTHONPATH": str(ROOT / "src")})
    for p in (ROOT / "src/go2_perception").rglob("*.py"):
        assert "go2_control" not in p.read_text(), p


TGT = MoveRequest(vx=0.30, vyaw=0.0, duration=7.0, resend_period=0.05, target=1.5)


def test_distance_target_stops_early_and_stops_robot():
    clock = FakeClock()
    progress = lambda: (clock.t - 100.0) * 0.3  # 0.3 m/s from the start
    out, route = run(req=TGT, clock=clock, progress=progress)
    assert out.end_reason.startswith("target reached")
    assert clock.t < 100.0 + 7.0
    assert route.calls[-4:] == ["stop", "stop", "stop", "release"]


def test_lost_progress_aborts_with_stop():
    clock = FakeClock()
    progress = lambda: None if clock.t > 101.0 else 0.1
    out, route = run(req=TGT, clock=clock, progress=progress)
    assert out.end_reason == "progress unavailable"
    first_stop = route.calls.index("stop")
    assert not any(isinstance(c, tuple) for c in route.calls[first_stop:])


def test_target_without_progress_or_before_move_is_refused():
    route = FakeRoute()
    with pytest.raises(Refused, match="progress"):
        run(req=TGT, route=route)
    assert route.calls == []
    route = FakeRoute()
    with pytest.raises(Refused, match="progress"):
        run(req=TGT, route=route, progress=lambda: None)
    assert not any(isinstance(c, tuple) or c == "claim" for c in route.calls)


@pytest.mark.parametrize(
    "req",
    [
        MoveRequest(0.3, 0.0, 7.0, 0.05, target=2.0),  # beyond MAX_TARGET
        MoveRequest(0.3, 0.0, 9.0, 0.05, target=1.5),  # beyond MAX_TARGET_DURATION
        MoveRequest(0.3, 0.0, 7.0, 0.0, target=1.5),  # target needs a stream
        MoveRequest(0.0, 0.3, 7.0, 0.05, target=2.0),  # heading target beyond ~90 deg
        MoveRequest(0.3, 0.0, 7.0, 0.05),  # long duration only with a target
        MoveRequest(0.3, 0.0, 1.0, 0.05, target=-1.0),
    ],
)
def test_invalid_targets_rejected(req):
    with pytest.raises(Refused):
        validate(req)


def test_reverse_only_with_target_and_speed_limit():
    validate(MoveRequest(-0.25, 0.0, 3.0, 0.05, target=0.3))
    for req in (
        MoveRequest(-0.1, 0.0, 1.0, 0.05),  # no measured stop
        MoveRequest(-0.80, 0.0, 3.0, 0.05, target=0.3),  # faster than the reverse limit
    ):
        with pytest.raises(Refused):
            validate(req)


def test_reverse_target_stops_on_progress():
    req = MoveRequest(-0.25, 0.0, 3.0, 0.05, target=0.3)
    clock = FakeClock()
    out, route = run(req=req, clock=clock, progress=lambda: (clock.t - 100.0) * 0.25)
    assert out.end_reason.startswith("target reached")
    assert all(c[1] == -0.25 for c in route.calls if isinstance(c, tuple))


def test_stall_stops_when_robot_stops_progressing():
    """R2-like: progress plateaus at 0.27 m while commands continue."""
    req = MoveRequest(-0.25, 0.0, 8.0, 0.05, target=1.5)
    clock = FakeClock()
    progress = lambda: min((clock.t - 100.0) * 0.1, 0.27)
    out, route = run(req=req, clock=clock, progress=progress)
    assert out.end_reason.startswith("stalled")
    assert clock.t - 100.0 < 6.0  # well before the 8 s cap
    assert route.calls[-4:] == ["stop", "stop", "stop", "release"]


def test_slow_steady_progress_is_not_a_stall():
    req = MoveRequest(0.30, 0.0, 8.0, 0.05, target=1.5)
    clock = FakeClock()
    out, _ = run(req=req, clock=clock, progress=lambda: (clock.t - 100.0) * 0.06)  # 0.09 m per window
    assert not out.end_reason.startswith("stalled")


def test_fuse_progress_rejects_lidar_jumps():
    pytest.importorskip("unitree_sdk2py")
    from go2_control.sdk_routes import fuse_progress

    assert fuse_progress(2.381, 0.053) == (0.053, False)  # X3 glitch
    assert fuse_progress(0.362, 0.294) == (0.362, True)  # C4: LiDAR accurate, odom short
    assert fuse_progress(None, 0.2) == (0.2, False)
    assert fuse_progress(0.4, None) == (0.4, True)
    assert fuse_progress(None, None) == (None, False)


def test_longer_stall_grace_lets_a_late_start_through_but_stays_bounded():
    req = MoveRequest(-0.25, 0.0, 8.0, 0.05, target=0.5)
    clock = FakeClock()
    late = lambda: max(0.0, (clock.t - 103.0) * 0.2)  # starts moving after 3 s
    out, _ = run(req=req, clock=clock, progress=late, stall_grace=5.0)
    assert out.end_reason.startswith("target reached")
    clock = FakeClock()
    out, _ = run(req=req, clock=clock, progress=late)  # default 2 s grace stops it
    assert out.end_reason.startswith("stalled")
    with pytest.raises(Refused, match="stall grace"):
        run(req=req, progress=late, stall_grace=7.0)


def test_heading_target_turns():
    validate(MoveRequest(0.0, -1.0, 3.0, 0.05, target=0.5236))
    for req in (
        MoveRequest(0.0, 1.0, 1.0, 0.05),  # fast turn without a heading stop
        MoveRequest(0.0, 1.2, 3.0, 0.05, target=0.5),  # faster than the turn limit
    ):
        with pytest.raises(Refused):
            validate(req)
    req = MoveRequest(0.0, -1.0, 3.0, 0.05, target=0.5236)
    assert confirmation_phrase(req).endswith("stop at 0.52rad")
    clock = FakeClock()
    out, route = run(req=req, clock=clock, progress=lambda: (clock.t - 100.0) * 0.6)
    assert out.end_reason.startswith("target reached")
    assert all(c[2] == -1.0 for c in route.calls if isinstance(c, tuple))
