"""Mode changes close the current offline session before the new session acquires its sources."""

from html import escape
from pathlib import Path
from threading import Event, Lock

from .interactive_server import DisplayServer


class WebsiteMode:
    names = {
        "observe": "Observation only",
        "five-move": "Supervised five-move · mock",
        "continuous": "Experimental continuous · offline chair simulation",
    }

    def __init__(self, name, port):
        self.name, self.port = name, port
        self.next_mode = None
        self.changing, self.stop = Event(), Event()
        self.lock = Lock()

    def server(self, app):
        if self.name == "observe":
            self.source_label = (
                f"{app.mode} camera · {app.run_info['backend']} predictions"
            )
        elif self.name == "five-move":
            self.source_label = "Synthetic camera and LiDAR · mock executor"
        else:
            self.source_label = "Simulated chair · synthetic geometry and planner"
        server = DisplayServer(app, self.port, application=self)
        self.port = server.httpd.server_address[1]
        return server

    def page(self, original):
        options = "".join(
            f'<option value="{key}"{" selected" if key == self.name else ""}>{escape(label)}</option>'
            for key, label in self.names.items()
        )
        bar = (
            Path(__file__)
            .with_name("mode_controls.html")
            .read_text()
            .replace("__OPTIONS__", options)
            .replace("__SOURCE__", escape(self.source_label))
        )
        return original.replace("<!-- application modes -->", bar, 1)

    def command(self, app, payload, now):
        action = payload.get("action")
        if action not in ("switch", "stop"):
            raise ValueError("Choose a mode or stop the current operation")
        with self.lock:
            if self.changing.is_set():
                raise ValueError("Previous mode is closing")
            if action == "switch" and payload.get("mode") not in self.names:
                raise ValueError("Unknown application mode")
            self.stop.set()
            if self.name == "five-move":
                app.navigation.enqueue({"action": "stop"}, now)
            elif self.name == "observe" and action == "stop":
                app.guided.enqueue({"action": "cancel"}, now)
            if action == "switch":
                self.next_mode = payload["mode"]
                self.changing.set()
