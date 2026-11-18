"""CPU-only UI/protocol regression checks; localhost only, no model or robot."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.error
import urllib.request

import pytest
import jsonschema

from go2_perception.vln.interactive_server import DisplayServer
from go2_perception.vln.supervised import Hold, current_checks, validate_record
from go2_perception.vln.supervised_app import NavigationMockWorker, navigation_page
from test_supervised_navigation import FIXTURE, setup_cycle
from test_vln_interactive import make_app

ROOT = Path(__file__).resolve().parents[1]


def test_ordinary_readonly_server_has_no_navigation_endpoint(tmp_path):
    app, clock, feed, messages = make_app(tmp_path)
    server = DisplayServer(app, 0)
    server.start()
    try:
        request = urllib.request.Request(server.url+"api/navigation", data=b'{"action":"authorize"}',
                                         headers={"Content-Type": "application/json", "X-Review-Token": server.token})
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        assert error.value.code == 404
        assert not hasattr(app, "navigation")
    finally:
        server.close()
        app.session.close(clock())


def test_supervised_page_binds_loaded_image_and_keeps_four_panels():
    page = navigation_page()
    assert all(f'id="{name}"' in page for name in ("cam", "pimg", "path", "lidar"))
    assert "Request STOP from executor" in page and "Cancel pending proposal" in page
    assert "navImageReady=img&&p.source_image?p.source_image.file_sha256:null" in page
    assert "p.prediction_id!==S.prediction.prediction_id)return" in page
    assert "panelNavigation();" in page and "strip();panelTest();" not in page


def test_mock_native_archive_equals_the_returned_canned_result(tmp_path):
    from types import SimpleNamespace
    request = SimpleNamespace(folder=tmp_path,
                              job={"episode_id": "synthetic", "clock_epoch": "synthetic",
                                   "decision_id": "1", "frames": []}, context={"identity": {}})
    worker = NavigationMockWorker(.1)
    worker.submit(request, 1.)
    response = worker.poll(1.2)
    assert json.loads((tmp_path / "mock_result.json").read_text()) == response["result"]


def test_unknown_or_nonfinite_geometry_does_not_mean_free(tmp_path):
    c, proposal, g, clock, first, second = setup_cycle(tmp_path)
    current = g.snapshot()
    current["geometry"]["forward_clear_m"] = float("nan")
    with pytest.raises(Hold, match="invalid_number"):
        current_checks(current, c.action, c.cap, clock(), capture=False)


def test_close_keeps_completed_result_state(tmp_path):
    c, proposal, g, clock, first, second = setup_cycle(tmp_path)
    c.state, c.reason = "COMPLETE", "saved"
    assert c.cancel() == "cancel" and c.state == "COMPLETE" and c.reason == "saved"


def test_mapping_action_specific_and_native_nearzero_chunk():
    from go2_perception.vln.supervised import plan
    cap, p = deepcopy(FIXTURE["capability"]), deepcopy(FIXTURE["proposal"])
    cap["mapping"]["verified_actions"] = ["forward"]
    p["waypoints"] = [[0, 0, .2]]*10
    with pytest.raises(Hold, match="mapping_not_verified_for_turn"):
        plan(p, cap)
    p["waypoints"] = [[0, 0, 0]]*10
    with pytest.raises(Hold, match="near_zero_chunk"):
        plan(p, cap)


def test_current_guard_future_stamp_is_refused(tmp_path):
    c, proposal, g, clock, first, second = setup_cycle(tmp_path)
    current = g.snapshot()
    current["pose"]["monotonic_s"] = clock()+.01
    with pytest.raises(Hold, match="future_robot_state"):
        current_checks(current, c.action, c.cap, clock(), capture=False)


def test_complete_browser_api_mock_cycle(tmp_path):
    output = tmp_path / "mock_cycle"
    run = subprocess.run([sys.executable, str(ROOT / "scripts/demo_vln_supervised_mock.py"), "--output", str(output)],
                         env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "GO2_HARDWARE": "0",
                              "GO2_SUPERVISED_LIVE": "0", "MUJOCO_RENDER_TEST": "0"},
                         capture_output=True, text=True, timeout=35)
    assert run.returncode == 0, run.stdout + run.stderr
    result = json.loads((output / "navigation/cycle_result.json").read_text())
    assert result["state"] == "COMPLETE" and result["result"]["synthetic"]
    validate_record("execution_result", result["result"])
    archive = json.loads((output / "navigation/cycle_record.json").read_text())
    schema = json.loads((ROOT / "src/go2_perception/vln/schemas/supervised-navigation-v1.schema.json").read_text())
    jsonschema.validate(archive, schema)
    assert result["result"]["measured_progress"] == pytest.approx(.08)
    assert result["post_observation"]["identity"]["observed_monotonic_s"] > result["result"]["ended_monotonic_s"]
    summary = json.loads((output / "summary.json").read_text())
    assert summary["robot_commands_sent"] == 0 and summary["supervised_navigation"]["state"] == "COMPLETE"
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    dispatch = next(e["ticket"] for e in events if e["event"] == "navigation_dispatch")
    auth = dispatch["authorization"]
    assert auth["created_monotonic_s"]-auth["reviewed_identity"]["observed_monotonic_s"] > 3
    assert dispatch["proposal"]["identity"]["observed_monotonic_s"] > auth["created_monotonic_s"]
    assert len([e for e in events if e["event"] == "navigation_dispatch"]) == 1
    assert summary["session_counts"]["submitted"] == 2  # one preview, one independent refresh
