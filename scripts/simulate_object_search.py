"""Five-move, step-and-wait pinhole simulation using the existing chair scene and offline planners."""

import argparse
from contextlib import closing, nullcontext
from copy import deepcopy
from html import escape
import json
import math
import os
from pathlib import Path
from tempfile import mkdtemp
from time import monotonic, sleep

from go2_perception.navigation import select_action
from go2_perception.vln.continuous_planner import LightNavPlanner, encode_view
from go2_perception.vln.continuous_simulation import CAMERA, START, fixture_result, render, sensor
from go2_perception.vln.episode_mock import effective_mock_capability
from go2_perception.vln.motion_profile import SENSOR, Hold, wrap
from review_target_grounding import marked_image, page


def simulate(out, planner=None):
    root = Path(__file__).resolve().parents[1]
    fixture = json.loads((root / "examples/supervised_navigation/synthetic_cycle.json").read_bytes())
    cap = effective_mock_capability(deepcopy(fixture["capability"]))
    cap["turn_scope"] = SENSOR
    template = deepcopy(fixture["proposal"])
    template["identity"]["instruction"] = "Go to the chair and stop."
    pose = dict(template["identity"]["pose"])
    grant = dict(origin=dict(pose), alignment_deg=15, area_radius_m=2, target="chair", operator="fixture")
    rows, views, inspected = [], [], []
    stamp = START

    def capture():
        image = render(pose)
        png, meta = encode_view(image)
        fid = f"sim-F{round((stamp - START) * 4):03}"
        if planner:
            planner.capture(stamp, png, meta["png_sha256"], fid)
        return fid, image, png

    fid, image, png = capture()
    aligned = False
    for seq in range(1, 7):  # Five actions, then one settled observation.
        guard = sensor(pose, stamp)
        t0 = monotonic()
        if planner:
            planner.submit(template, guard, fid, CAMERA, (stamp, t0))
            delivery = None
            while delivery is None and monotonic() - t0 < 45:
                delivery = planner.poll(stamp + monotonic() - t0)
                if delivery is None:
                    sleep(0.01)
            if delivery is None:
                raise TimeoutError("synthetic scene inference timed out")
        else:
            delivery = fixture_result(template, guard, image, seq, 0)
        latency = monotonic() - t0
        proposal, evidence = delivery["proposal"], delivery["grounding"]
        image_path = out / f"decision-{seq:02}.png"
        image_path.write_bytes(png)
        inspected.append(dict(pose))
        row = dict(
            decision=seq,
            capture_s=stamp,
            frame_id=fid,
            pose=dict(pose),
            image=str(image_path),
            visible_in_renderer=bool(image.any()),
            point=proposal["point_fraction"],
            native_output=proposal["native_output"],
            native_stop=proposal["native_stop"],
            grounding=evidence["status"],
            latency_s=latency,
            path=proposal["waypoints"],
        )
        rows.append(row)
        if proposal["native_stop"]:
            outcome = "STOP_ARRIVAL_UNVERIFIED"
            break
        if len(views) == 5:
            outcome = "MOVE_BUDGET_EXHAUSTED"
            break
        try:
            selected = select_action(
                grant, cap, proposal, evidence, views, len(views), aligned, inspected=inspected
            )
        except Hold as exc:
            outcome = str(exc)
            break
        action = selected.get("action")
        row["selection"] = selected
        if not action:
            outcome = selected["reason"]
            break
        aligned = selected["aligned"]
        direction = action["direction"]
        views.append(dict(pose, attempt=dict(mode=selected["mode"], direction=direction)))
        remaining = action["magnitude"]
        # Move only the ideal proxy. Capture intermediate views at its existing 4 Hz cadence.
        while remaining > 1e-9:
            step = min(remaining, 0.25 * action["speed"])
            if direction == "forward":
                pose["x_m"] += step * math.cos(pose["yaw_rad"])
                pose["y_m"] += step * math.sin(pose["yaw_rad"])
            else:
                pose["yaw_rad"] = wrap(pose["yaw_rad"] + (step if direction == "left" else -step))
            remaining -= step
            stamp += 0.25
            pose["monotonic_s"] = stamp
            fid, image, png = capture()
        for _ in range(4):
            stamp += 0.25
            pose["monotonic_s"] = stamp
            fid, image, png = capture()
    report = dict(
        backend="LightNav" if planner else "scripted pixel fixture",
        synthetic=True,
        instruction=template["identity"]["instruction"],
        moves=len(views),
        outcome=outcome,
        goal_evaluation="arrival unmeasured",
        decisions=rows,
    )
    (out / "simulation.json").write_text(json.dumps(report, indent=2) + "\n")
    panels = []
    for row in rows:
        sel = row.get("selection", {})
        act = sel.get("action") or {}
        panels.append(
            f"<article><h2>Decision {row['decision']}</h2>"
            + marked_image(row["image"], row["point"])
            + f"<p>{escape(sel.get('mode', ''))} {escape(act.get('direction', outcome))}</p>"
            + f"<p>Chair drawn: {row['visible_in_renderer']} · native STOP: {row['native_stop']}</p></article>"
        )
    page(
        out / "index.html",
        report["backend"] + " · closed-loop chair simulation",
        "<p>A settled kinematic proxy generates each next view. The synthetic image-region check is the "
        "only fixture identity evidence. Arrival is unmeasured.</p><div class='grid'>"
        + "".join(panels)
        + "</div>",
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("fixture", "lightnav"), required=True)
    args = parser.parse_args()
    workspace = Path(os.environ["GO2_WORKSPACE"]).resolve()
    base = workspace / "runs/target_grounding"
    base.mkdir(exist_ok=True, parents=True)
    out = Path(mkdtemp(dir=base, prefix="object-sim-" + args.backend + "-"))
    print(out, flush=True)
    with (
        closing(LightNavPlanner.start(out / "inference", workspace))
        if args.backend == "lightnav"
        else nullcontext(None)
    ) as planner:
        report = simulate(out, planner)
    print(report["outcome"], report["moves"], out / "index.html")


if __name__ == "__main__":
    main()
