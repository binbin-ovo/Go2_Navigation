"""Small local HTML/status display. No server, commands, path-on-camera registration."""

import html
import json
import math
from pathlib import Path
from urllib.parse import quote

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .shadow import write_json


def escaped(value):
    return html.escape(str(value), quote=True)


def relative(path, root):
    return quote(str(Path(path).resolve().relative_to(Path(root).resolve())))


def panel_text(draw, xy, text, *, fill, size):
    for name in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans.ttf",
    ):
        try:
            font = ImageFont.truetype(name, size)
            break
        except OSError:
            continue
    else:
        font = ImageFont.load_default(size=size)
    draw.text(xy, str(text), fill=fill, font=font)


def point_fraction(token):
    """Pinned vln_utils.decode_point_pixel_center, 48x27 cells, image fraction.

    Edge cells may represent clamped predictions, not verified object locations.
    """
    if type(token) is not int or not 1 <= token <= 1296:
        return None
    return (((token - 1) % 48 + 0.5) / 48, ((token - 1) // 48 + 0.5) / 27)


def target_fraction(pointing):
    """Pinned grid/1000-bin target channel; conflicting formats stay textual."""
    if not isinstance(pointing, dict):
        return None
    grid = point_fraction(pointing.get("opos"))
    posxy, sentinels = pointing.get("posxy", {}), pointing.get("sentinels", {})
    if not isinstance(posxy, dict) or not isinstance(sentinels, dict):
        return None
    pair = posxy.get("opos")
    if pair is not None:
        if grid is not None or sentinels.get("opos") is not None:
            return None
        if (
            not isinstance(pair, (tuple, list))
            or len(pair) != 2
            or any(type(v) is not int or not 0 <= v < 1000 for v in pair)
        ):
            return None
        return tuple((v + 0.5) / 1000 for v in pair)
    return grid


def path_svg(chunk):
    try:
        w = np.asarray(chunk, dtype=float)
        if w.shape != (10, 3) or not np.isfinite(w).all():
            return "<p>No valid native chunk.</p>"
    except (TypeError, ValueError):
        return "<p>No valid native chunk.</p>"
    # Robot-forward up, left left. This is a separate local chart, never an overlay.
    scale = 115 / max(0.25, float(np.abs(w[:, :2]).max()))
    coords = [(220 - y * scale, 170 - x * scale) for x, y, _ in w]
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in [(220, 170), *coords])
    arrows = []
    for i, ((x, y), (_, _, yaw)) in enumerate(zip(coords, w)):
        dx, dy = -math.sin(yaw) * 13, -math.cos(yaw) * 13
        arrows.append(
            f'<circle cx="{x}" cy="{y}" r="3" fill="#127eae"/><path d="M{x},{y} l{dx},{dy}" stroke="#b34c12"/><text x="{x + 4}" y="{y - 4}" font-size="10">{i}</text>'
        )
    return f'''<svg viewBox="0 0 440 330" role="img" aria-label="Full native waypoint chunk in local decision frame">
    <rect width="440" height="330" fill="#f1f6f7"/><path d="M220,300V20 M25,170H415" stroke="#aab8bc"/>
    <polyline points="{points}" fill="none" stroke="#127eae" stroke-width="2"/>{"".join(arrows)}
    <circle cx="220" cy="170" r="5" fill="#111"/><text x="230" y="24">+x forward</text>
    <text x="25" y="160">+y left</text><text x="15" y="318">yaw CCW; origin = observation decision frame; metres/radians</text></svg>'''


def render(session, now, *, snapshot=False):
    root = session.output
    state = session.status(now)
    write_json(root / "status.json", state)
    latest, shown = state["latest"], state["displayed"]
    result = {} if not shown else shown["response"].get("result", {})
    label = "WAITING FOR PROPOSAL" if not shown else shown["label"]
    backend = (
        "MOCK — no model inference"
        if result.get("mock")
        else "LightNav"
        if shown
        else "waiting"
    )
    banner = state["banner"]
    latest_info = "No camera observation"
    if latest:
        latest_info = f"{latest['frame_id']} · {latest['observation_s']:.6f} s · {latest['stamp']['clock']} ({latest['stamp']['semantics']}) · {latest['clock_epoch']}"
    live_html = ""
    if state.get("source_status") is not None:
        live = state["source_status"]
        receipt_age = live.get("host_since_last_sensor_receipt_s")
        exposure_age = (
            None
            if not latest or latest.get("exposure_monotonic_s") is None
            else now - latest["exposure_monotonic_s"]
        )
        camera = live.get("camera_state", "unknown")
        live_html = f"""<section><h2>Live camera · {escaped(camera.upper())}</h2>
    <p>Latest frame: host time since sensor receipt {escaped(None if receipt_age is None else f"{receipt_age:.3f} s")} ·
    since estimated exposure {escaped("UNKNOWN" if exposure_age is None else f"{exposure_age:.3f} s (estimated, unvalidated)")}.
    An interruption is reported here; old frames are never re-fed.</p><pre>{escaped(json.dumps({k: live.get(k) for k in ("counts", "clock_epoch", "bridge_alive", "bridge_error")}, indent=2))}</pre></section>"""
    model_view = "<p>No model input yet.</p>"
    history_html, timing_text, proposal_instruction = "", "Not inferred", ""
    current_path = None
    if shown:
        folder = Path(shown["request_directory"])
        job = json.loads((folder / "job.json").read_text())
        proposal_instruction = job["instruction"]
        indices = result.get("native_selected_frame_indices", [])
        if not isinstance(indices, list):
            indices = []
        for i in indices:
            if type(i) is not int or not 0 <= i < len(job["frames"]):
                continue
            native = folder / "native" / f"input_{i:04d}.png"
            path = Path(job["frames"][i]["path"]) if result.get("mock") else native
            if not path.is_file():
                continue
            stamp = job["frames"][i]["observation_s"]
            history_html += f'<figure><img src="{relative(path, root)}"><figcaption>input {i} · {stamp:.6f} s</figcaption></figure>'
            current_path = path
        if current_path:
            overlay = ""
            point = target_fraction(result.get("pointing", {}))
            if point:
                overlay = f'<span class="target" style="left:{100 * point[0]}%;top:{100 * point[1]}%" title="Native target point: not verified">+</span>'
            model_view = f'<div class="model-image"><img src="{relative(current_path, root)}">{overlay}</div>'
        age = shown["display_timing"]
        timing_text = json.dumps(
            {
                **age,
                "inference_duration_s": shown["response"].get("inference_duration_s"),
                "native_generation_s": result["timing"].get("native_generate_s")
                if isinstance(result.get("timing"), dict)
                else None,
                "host_inference_start_s": shown["response"].get(
                    "host_inference_start_s"
                ),
                "host_inference_end_s": shown["response"].get("host_inference_end_s"),
                "rejection": shown["current_rejection_reason"],
            },
            indent=2,
        )
    chunk = result.get("waypoints")
    rows = ""
    if isinstance(chunk, list) and len(chunk) == 10:
        for i, row in enumerate(chunk):
            if isinstance(row, list) and len(row) == 3:
                rows += (
                    "<tr>"
                    + "".join(f"<td>{escaped(v)}</td>" for v in [i, *row])
                    + "</tr>"
                )
    source = state["source_mode"].upper()
    if "LIVE" in source:
        source = "LIVE INPUT — UNVALIDATED"
    elif "SYNTHETIC" in source:
        source = "REPLAY / " + source
    else:
        source = "REPLAY / " + source
    content = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta http-equiv="refresh" content="1">
    <title>Go2 read-only VLN advisory</title><style>
    *{{box-sizing:border-box}}body{{margin:0;background:#e8edf1;color:#142b39;font:15px system-ui,sans-serif}}
    header{{background:#173b50;color:white;padding:18px 28px}}header strong{{font-size:23px}}main{{padding:18px 28px;max-width:1600px;margin:auto}}
    .badge{{background:#ffdc88;color:#362b13;padding:5px 10px;border-radius:6px;display:inline-block;margin:8px 8px 0 0}}
    .grid{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px}}section{{background:white;padding:16px;border-radius:10px;margin-bottom:16px;overflow:auto}}
    h2{{font-size:17px;margin:0 0 10px}}img{{width:100%;object-fit:contain}}figure{{margin:0}}figcaption,small{{color:#486170}}
    pre{{white-space:pre-wrap;word-break:break-word;font-size:12px}}table{{font:12px monospace;width:100%}}td{{padding:4px}}
    .history{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.model-image{{position:relative}}.target{{position:absolute;transform:translate(-50%,-50%);color:#ff32cd;font-size:32px;font-weight:bold}}
    @media(max-width:900px){{.grid{{grid-template-columns:1fr}}.history{{grid-template-columns:1fr 1fr}}}}
    </style><header><strong>{escaped(banner)}</strong><br><span class="badge">{escaped(source)}</span><span class="badge">{escaped(backend)}</span><span class="badge">{escaped(label)}</span></header><main>
    <section><h2>Current instruction · revision {state["instruction_revision"]}</h2><p>{escaped(state["instruction"])}</p>
    <small>Session {escaped(state["session_id"])} · active={state["active"]} · in flight={state["inference_in_flight"]} · pending={state["pending_requests"]} · history={state["history_frames"]}</small>
    <p>{escaped(state["history_error"] or "")}</p><p>Terminal: <code>instruction NEW TEXT</code> · <code>cancel</code> · <code>status</code> · <code>quit</code>. Cancel cancels the advisory request only; it does not stop the robot and sends nothing to it.</p></section>{live_html}
    <div class="grid"><section><h2>Latest camera view — may be newer</h2><img src="latest.png"><p>{escaped(latest_info)}</p>
    <p>Pose: {"available (not registered)" if latest and latest["pose_available"] else "missing"} · LiDAR: {"separate observation only" if latest and latest["lidar_available"] else "missing"}</p></section>
    <section><h2>Image actually used by this proposal</h2>{model_view}<p>Proposal instruction: {escaped(proposal_instruction)}</p>
    <p>{"Mock rectified view; official processor not run" if result.get("mock") else "Actual official model-space input; target marker only when a native OPOS point exists."}</p>
    <p>Source cutoff: {escaped(shown["snapshot"]["source_cutoff_s"] if shown else None)} s. Exact input timestamps below.</p></section>
    <section><h2>Full native local waypoint chunk</h2>{path_svg(chunk)}<p>No Go2 base registration or intrinsic time base. This chart is not placed on the latest camera image.</p></section></div>
    <div class="grid"><section><h2>Native evidence</h2><pre>{escaped(result.get("native_output", "No output"))}</pre>
    <p>Native STOP: {escaped(result.get("decoder_stop"))} · explicit RVQ STOP: {escaped(result.get("explicit_stop_code"))}</p>
    <p>Model completion claim: {escaped((shown or {}).get("semantics", {}).get("model_completion_claim") if shown and shown.get("semantics") else None)}</p>
    <p>Verified goal completion: UNKNOWN. Physical robot stopping: UNKNOWN.</p><pre>{escaped(json.dumps(result.get("pointing", {}), indent=2))}</pre></section>
    <section><h2>Timing and validity</h2><pre>{escaped(timing_text)}</pre><p>Age limits are advisory display settings, not physical safety limits. Unknown capture age stays unknown.</p></section>
    <section><h2>All predicted poses</h2><table><tr><th>row</th><th>x forward m</th><th>y left m</th><th>yaw CCW rad</th></tr>{rows}</table></section></div>
    <section><h2>Exact current/history inputs for the displayed proposal</h2><p>All saved native-selected indices; preprocessing/tubelet details remain in the request artifacts.</p><div class="history">{history_html}</div></section>
    <p>T_base_camera remains unresolved. No measured target distance, fused semantic map, traversability or navigation-success claim. Model STOP is not proof of goal completion.</p></main></html>"""
    temp = root / "index.tmp.html"
    temp.write_text(content)
    temp.replace(root / "index.html")
    if snapshot:
        # Lightweight shareable panel; full history/chunk/timing remain in HTML/JSON.
        panel = Image.new("RGB", (1400, 850), "#e8edf1")
        draw = ImageDraw.Draw(panel)
        draw.rectangle((0, 0, 1400, 110), fill="#173b50")
        panel_text(draw, (24, 15), banner, fill="white", size=28)
        panel_text(
            draw, (24, 62), f"{source} | {backend} | {label}", fill="#ffdc88", size=20
        )
        panel_text(
            draw,
            (24, 125),
            f"Instruction r{state['instruction_revision']}: {state['instruction']}",
            fill="#142b39",
            size=19,
        )
        for path, x, title in (
            (root / "latest.png", 24, "Latest observation"),
            (current_path, 710, "Exact proposal input (older timestamps retained)"),
        ):
            panel_text(draw, (x, 170), title, fill="#142b39", size=18)
            if path and path.is_file():
                with Image.open(path) as img:
                    img.thumbnail((650, 380))
                    panel.paste(img, (x, 205))
        text = [
            latest_info,
            f"Native STOP={result.get('decoder_stop')} | completion UNVERIFIED | no physical stop command",
            "Native output: " + str(result.get("native_output", "none")),
            "Full native path, yaw, input history, timing and rejection reasons: index.html / status.json",
            ("Live read-only camera input. " if live_html else "Replay follows recorded/synthetic inputs. ") + "No proposal has been executed.",
        ]
        for i, line in enumerate(text):
            panel_text(draw, (24, 620 + i * 35), line[:145], fill="#142b39", size=17)
        if shown:
            panel_text(
                draw,
                (710, 580),
                f"Proposal source stamp: {shown['snapshot']['current_source']['observation_s']:.6f} s",
                fill="#142b39",
                size=17,
            )
        panel.save(root / "advisory.png")
    return state
