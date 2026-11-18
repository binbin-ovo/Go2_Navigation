"""Loopback-only display server for the interactive review tool (standard library only).

Binds 127.0.0.1. GETs return snapshots; the only POSTs are the browser's report of
what it is displaying (for exact mark binding) and REVIEW-mode target corrections,
both requiring a per-run token. No endpoint can reach the model, the sensors' bridges
or the robot. Host headers other than localhost are refused (DNS rebinding).
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import secrets
import threading
import time
from urllib.parse import parse_qs, urlparse

import cv2

from .interactive_page import PAGE

ALLOWED_HOSTS = ("127.0.0.1", "localhost")


def make_handler(app, token, monotonic=time.monotonic):
    class Handler(BaseHTTPRequestHandler):
        server_version = "Go2VLNReview/1"

        def log_message(self, *args):  # keep the operator's terminal clean
            pass

        def _host_ok(self):
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
            return host in ALLOWED_HOSTS

        def _send(self, code, body, kind="application/json"):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, allow_nan=False, default=str).encode()
            elif isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self._host_ok():
                return self._send(403, {"error": "loopback host required"})
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            now = monotonic()
            try:
                if url.path == "/":
                    return self._send(200, PAGE.replace("__TOKEN__", token), "text/html; charset=utf-8")
                if url.path == "/api/state":
                    return self._send(200, _finite(app.state(now)))
                if url.path == "/api/camera":
                    if app.tap is None:
                        return self._send(404, {"error": "no camera"})
                    frame_id, data = app.tap.latest_jpeg(q.get("f"))
                    if data is None:
                        return self._send(404, {"error": "frame not in display ring"})
                    return self._send(200, data, "image/jpeg")
                if url.path == "/api/lidar":
                    if app.lidar is None:
                        return self._send(404, {"error": "no lidar"})
                    view = app.lidar.view(q.get("view", "latest"))
                    return self._send(200, {"view": q.get("view", "latest"), "points": view})
                if url.path == "/api/pred_image":
                    path = app.prediction_image_path(q.get("id"))
                    if path is None:
                        return self._send(404, {"error": "unknown prediction image"})
                    return self._send(200, path.read_bytes(), "image/png")
                if url.path == "/api/mark_image":
                    image = app.mark_image(q.get("trial"), q.get("mark"))
                    if image is None:
                        return self._send(404, {"error": "unknown mark image"})
                    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    return self._send(200, buf.tobytes(), "image/jpeg")
            except (ValueError, OSError, KeyError) as error:
                return self._send(400, {"error": str(error)})
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            if not self._host_ok() or not secrets.compare_digest(self.headers.get("X-Review-Token", ""), token):
                return self._send(403, {"error": "token and loopback host required"})
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 < length <= 4096:
                return self._send(400, {"error": "small JSON body required"})
            try:
                payload = json.loads(self.rfile.read(length))
                now = monotonic()
                if self.path == "/api/viewing":
                    app.report_view(payload, now)
                    return self._send(200, {"ok": True})
                if self.path == "/api/guided":
                    app.guided.enqueue(payload, now)
                    return self._send(200, {"ok": True, "queued": payload.get("action")})
                if self.path == "/api/correction":
                    record = app.correction(payload, now)
                    return self._send(200, {"ok": True, "correction_id": record["correction_id"]})
            except (ValueError, KeyError, TypeError) as error:
                return self._send(400, {"error": str(error)})
            return self._send(404, {"error": "not found"})

    return Handler


def _finite(value):
    """Replace non-finite floats (unknown) with None so the page shows UNKNOWN, not NaN."""
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


class DisplayServer:
    def __init__(self, app, port=8765):
        self.token = secrets.token_urlsafe(16)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app, self.token))
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True, name="display-server")

    @property
    def url(self):
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/"

    def start(self):
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)
