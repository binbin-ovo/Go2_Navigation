"""Exercise the three public modules with offline inputs and an actual loopback server."""

import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from go2_perception import navigation
from go2_perception.sensors import SensorSettings, open_sensors
from go2_perception.vln import episode_policy

ROOT = Path(__file__).resolve().parents[1]


def until(read, timeout=10):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = read()
        if value:
            return value
        time.sleep(0.04)
    pytest.fail("Timed out waiting for " + read.__name__)


def test_public_source_and_policy_have_one_implementation():
    from go2_perception import sensors

    for module in (navigation, sensors):
        assert (ROOT / Path(module.__file__).name).samefile(module.__file__)
    assert navigation.select_action is episode_policy.select_action
    assert (
        navigation.target_spec("Go to the large brown cardboard box and stop.")[
            "operator_text"
        ]
        == "large brown cardboard box"
    )


@pytest.mark.parametrize(
    "camera,lidar", [("episode", "go2"), ("go2", "recording"), ("typo", "none")]
)
def test_unsupported_source_pairs_refused_before_io(tmp_path, camera, lidar):
    with pytest.raises(ValueError):
        open_sensors(
            SensorSettings(
                camera=camera, lidar=lidar, hardware_python=Path(sys.executable)
            ),
            tmp_path,
        )
    assert not list(tmp_path.iterdir())


def test_public_read_only_sensor_lifecycle_and_clock(tmp_path):
    sensors = open_sensors(SensorSettings(lidar="synthetic"), tmp_path)
    assert sensors.latest_frame() is None and not sensors.pose()["valid"]
    try:
        sensors.start()
        sensors.start()
        rgb, meta = until(sensors.latest_frame)
        scan, timestamp = until(sensors.latest_cloud)
        pose = until(lambda: (p if (p := sensors.pose())["valid"] else None))
        assert rgb.ndim == 3 and scan.ndim == 2 and scan.shape[1] == 3
        assert meta["stamp_host"] and meta["received_monotonic_s"] <= time.monotonic()
        assert (
            timestamp["frame_id"]
            and timestamp["robot_epoch"]
            and timestamp["stamp"] <= timestamp["t_mid"]
        )
        assert pose["frame"].startswith("odom") and np.isfinite(pose["yaw_rad"])
        frame_id = meta["frame_id"]
        original = sensors.tap.frame(frame_id)[0].copy()
        rgb.fill(0)
        assert np.array_equal(original, sensors.tap.frame(frame_id)[0])
        assert sensors.status()["lidar"]["bridge_alive"]
    finally:
        sensors.stop()
        sensors.stop()
    assert not sensors.status()["camera"]["alive"] and not sensors.lidar.alive


def test_public_replay_uses_recorded_stamps(tmp_path):
    episode = ROOT / "tests/fixtures/vln/episode_v1/episode.json"
    sensors = open_sensors(SensorSettings(camera="episode", episode=episode), tmp_path)
    try:
        sensors.start()
        _, frame = until(sensors.latest_frame)
        assert frame["clock_epoch"] and frame["stamp_host"] and frame["source_id"]
        assert sensors.calibration["episode_manifest"] == str(episode)
        assert sensors.status()["lidar"] is None and sensors.latest_cloud() is None
    finally:
        sensors.stop()


def test_website_switch_stop_and_session_isolation(tmp_path):
    log = tmp_path / "console.log"
    # Launch just as the owner would, even if the shared interpreter's editable points elsewhere.
    env = {
        **os.environ,
        "GO2_HARDWARE": "0",
        "GO2_SUPERVISED_LIVE": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    env.pop("PYTHONPATH", None)
    with log.open("w") as stream:
        child = subprocess.Popen(
            [
                sys.executable,
                "-u",
                ROOT / "app.py",
                "--port",
                "0",
                "--workspace",
                tmp_path,
            ],
            cwd=tmp_path,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    try:

        def started():
            if child.poll() is not None:
                pytest.fail(log.read_text())
            return re.search(r"Display: (http://127\.0\.0\.1:\d+/)", log.read_text())

        url = until(started)[1]

        def get(route):
            with urlopen(url + route, timeout=2) as response:
                return json.load(response)

        def command(token, action, mode=None, route="api/application", **fields):
            req = Request(
                url + route,
                data=json.dumps(dict(action=action, mode=mode, **fields)).encode(),
                headers={"Content-Type": "application/json", "X-Review-Token": token},
            )
            with urlopen(req, timeout=2) as response:
                return json.load(response)

        def wait_mode(name):
            def available():
                try:
                    state = get("api/application")
                except (URLError, ConnectionError):
                    return None
                return state if state["mode"] == name else None

            return until(available)["session"]

        first = wait_mode("observe")
        state = until(
            lambda: (
                s if (s := get("api/state")).get("camera", {}).get("latest") else None
            )
        )
        assert state["robot_commands_sent"] == 0 and state["trial"] is None
        with pytest.raises(HTTPError) as exc:
            command(first, "switch", "live")
        assert exc.value.code == 400
        with pytest.raises(HTTPError) as exc:
            command("invalid", "switch", "five-move")
        assert exc.value.code == 403
        command(first, "switch", "five-move")
        second = wait_mode("five-move")
        state = get("api/state")
        assert (
            state["navigation"]["state"] == "IDLE" and state["robot_commands_sent"] == 0
        )
        with pytest.raises(HTTPError) as exc:
            command(first, "authorize", route="api/navigation")
        assert exc.value.code == 403 and first != second
        command(second, "stop")
        until(lambda: get("api/state")["navigation"]["state"] == "TERMINAL")
        command(second, "switch", "continuous")
        third = wait_mode("continuous")
        command(third, "stop")
        result = until(
            lambda: next(
                tmp_path.glob("runs/application/site-*/003-continuous/summary.json"),
                None,
            )
        )
        summary = json.loads(result.read_bytes())
        assert (
            summary["physical_commands_sent"] == 0
            and summary["results"][0]["end_reason"] == "cancelled"
        )
        tick = get("api/continuous?tick=latest")
        assert tick["packet"]["velocity"] == [0.0, 0.0] and tick["geometry"][
            "source"
        ].startswith("synthetic")
        command(third, "switch", "observe")
        assert wait_mode("observe") != first
        assert not get("api/state").get("prediction")
    finally:
        if child.poll() is None:
            child.send_signal(signal.SIGINT)
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
            pytest.fail("Application did not close after SIGINT")
    assert child.returncode == 0, log.read_text()


def test_public_imports_from_project_root_use_this_worktree():
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("PYTHONPATH", None)
    result = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import sensors, navigation, go2_perception; print(go2_perception.__file__)",
        ],
        cwd=ROOT,
        env=env,
        text=True,
    )
    assert Path(result.strip()).resolve().is_relative_to(ROOT / "src")


@pytest.mark.parametrize(
    "script", ["launch_five_move_episode.sh", "launch_effective_trial.sh"]
)
def test_published_live_launcher_refuses_missing_opt_in(tmp_path, script):
    result = subprocess.run(
        ["bash", ROOT / "scripts" / script, "--live-exclusive"],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "GO2_WORKSPACE": str(tmp_path),
            "GO2_HARDWARE": "0",
            "GO2_SUPERVISED_LIVE": "0",
        },
    )
    assert result.returncode == 2 and "exclusive command ownership" in result.stderr
    assert not list(tmp_path.iterdir())


def episode_signature(folder):
    """Compare stable data across fresh nonces, independently paced captures and dispatch clocks."""
    import hashlib
    import cv2
    from go2_perception.vln.grounding_review import episode_decisions
    from go2_perception.vln.prediction_archive import records

    episode, decisions = episode_decisions(folder)
    predictions = records(next((folder / "trials").glob("T*/predictions.jsonl")))
    observed = []
    for p in predictions:
        source = p["source_image"]
        current = cv2.imread(str(folder / source["path"]), cv2.IMREAD_COLOR)
        ev = decisions[p["prediction_id"]]
        action = ev["selection"]
        outcome = ev["result"]
        observed.append(
            {
                "instruction": p["instruction"],
                "current_bgr_sha256": hashlib.sha256(current.tobytes()).hexdigest(),
                "native_output": p["native_output"],
                "native_stop": p["native_stop"],
                "waypoints_sha256": hashlib.sha256(
                    json.dumps(p["waypoints"], separators=(",", ":")).encode()
                ).hexdigest(),
                "grounding": ev["grounding"]["status"],
                "selection": None
                if action is None
                else [
                    action["mode"],
                    action["source"],
                    action["action"]["direction"],
                    action["action"]["magnitude"],
                    action["action"]["unit"],
                ],
                "cleanup": None
                if outcome is None
                else {
                    k: outcome[k]
                    for k in (
                        "status",
                        "observed_stationary",
                        "ownership_released",
                        "post_release_stationary",
                    )
                },
            }
        )
    result = json.loads((folder / "summary.json").read_bytes())
    return dict(
        decisions=observed,
        moves=episode["moves"],
        outcome=episode["outcome"],
        physical_commands=result["robot_commands_sent"],
        physical_authorization=result["physical_execution_authorized"],
    )


def test_mock_episode_matches_publication_before_refactor(tmp_path):
    session = tmp_path / "episode"
    subprocess.run(
        [
            sys.executable,
            ROOT / "scripts/demo_vln_supervised_mock.py",
            "--five-move-episode",
            "--output",
            session,
        ],
        cwd=ROOT,
        check=True,
        timeout=40,
    )
    reference = json.loads(
        (ROOT / "tests/fixtures/supervised/five_move_behavior.json").read_bytes()
    )
    assert episode_signature(session) == reference["expected"]


def test_public_operator_entry_needs_explicit_authorization(tmp_path):
    result = subprocess.run(
        [sys.executable, ROOT / "app.py", "supervised", "--live-exclusive"],
        cwd=ROOT,
        env={
            **os.environ,
            "GO2_WORKSPACE": str(tmp_path),
            "GO2_HARDWARE": "0",
            "GO2_SUPERVISED_LIVE": "0",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2 and "exclusive command ownership" in result.stderr
    assert not list(tmp_path.iterdir())
