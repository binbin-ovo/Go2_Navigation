"""Drive ONE synthetic cycle through the real loopback UI/worker/executor IPC.

No trained model, GPU, robot SDK or sensor connection. Target identity, scene and
capabilities are invented. This tests software integration, not navigation quality.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--five-move-episode", action="store_true")
    a = p.parse_args(argv)
    if a.output.exists():
        p.error("fresh output directory required")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "GO2_HARDWARE": "0", "GO2_SUPERVISED_LIVE": "0",
           "MUJOCO_RENDER_TEST": "0", "PYTHONDONTWRITEBYTECODE": "1"}
    process = subprocess.Popen([sys.executable, "-u", str(ROOT / "scripts/run_vln_supervised.py"),
                                *(["--five-move-episode", "--profile", "go2-effective-v1"] if a.five_move_episode else []),
                                "--lidar", "synthetic", "--output", str(a.output), "--port", "0", "--max-host-seconds", "30",
                                "--min-free-disk-gb", "0", "--min-available-memory-gb", "0"],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1, env=env)
    lines, found = [], threading.Event()
    address = []

    def read():
        for line in process.stdout:
            lines.append(line)
            match = re.search(r"Display: (http://127\.0\.0\.1:\d+/)", line)
            if match:
                address.append(match[1])
                found.set()

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    states, status = [], "failed"
    token = url = None

    def get(path):
        return urllib.request.urlopen(url+path, timeout=3).read().decode()

    def send(payload):
        request = urllib.request.Request(url+"api/navigation", data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json", "X-Review-Token": token})
        with urllib.request.urlopen(request, timeout=3) as response:
            return json.load(response)

    def wait(predicate, timeout=8):
        end = time.monotonic()+timeout
        while time.monotonic() < end:
            state = json.loads(get("api/state"))
            if not states or states[-1]["navigation"] != state["navigation"]:
                states.append({"monotonic_s": time.monotonic(), "navigation": state["navigation"]})
            if predicate(state):
                return state
            time.sleep(.05)
        raise RuntimeError("mock integration timeout: " + json.dumps(states[-1] if states else {}))

    try:
        if not found.wait(8):
            raise RuntimeError("viewer did not start: " + "".join(lines))
        url = address[0]
        page = get("")
        token = re.search(r'const TOKEN="([^"]+)"', page).group(1)
        if a.five_move_episode:
            wait(lambda s: s.get("camera", {}).get("latest") and (s.get("model") or {}).get("state") == "READY")
            # Both camera and executor must have had their first independent guard sample.
            send({"action":"authorize", "target":"go to the chair", "operator":"synthetic operator",
                  "area_inspected":True, "exclusive_and_intervention_confirmed":True})
            final = wait(lambda s: s["navigation"]["state"] == "TERMINAL", timeout=18)["navigation"]
            fixture = json.loads((ROOT/"examples/five_move_navigation/synthetic_episode.json").read_text())
            if (final.get("outcome"), final["moves"]) != (fixture["expected_outcome"], fixture["expected_moves"]):
                raise RuntimeError(json.dumps(final))
            status = "completed_synthetic_episode"
            return 0
        send({"action": "ask", "target": "synthetic textured panel"})
        state = wait(lambda s: s.get("prediction") is not None)
        pred = state["prediction"]
        binding = {"prediction_id": pred["prediction_id"], "source_image_sha256": pred["source_image"]["file_sha256"]}
        send({"action": "confirm", **binding, "roi": [.2, .2, .8, .8],
              "instance": "invented-panel-instance-1", "operator": "SYNTHETIC automated operator"})
        wait(lambda s: s["navigation"]["state"] == "REVIEW")
        time.sleep(3.25)  # deliberately let review exceed the unchanged 3-s inference-age limit
        send({"action": "authorize", **binding, "safety_confirmed": True})
        final = wait(lambda s: s["navigation"]["state"] in ("COMPLETE", "HOLD"), timeout=12)
        if final["navigation"]["state"] != "COMPLETE":
            raise RuntimeError(final["navigation"]["reason"])
        status = "completed_synthetic_software_cycle"
        return 0
    finally:
        if url and token:
            try:
                send({"action": "quit"})
            except (OSError, ValueError):
                pass
        process.stdin.close()
        process.wait(timeout=20)
        reader.join(timeout=1)
        if a.output.exists():
            (a.output / "mock_demo_driver.json").write_text(json.dumps({"status": status, "synthetic": True,
                                                                      "model_loaded": False, "states": states}, indent=2)+"\n")
            (a.output / "mock_demo_console.txt").write_text("".join(lines))
        print(json.dumps({"status": status, "output": str(a.output.resolve()), "child_exit": process.returncode}))


if __name__ == "__main__":
    raise SystemExit(main())
