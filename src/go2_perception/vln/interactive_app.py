"""Interactive four-panel VLN review and data collection. ADVISORY ONLY — NO ROBOT COMMANDS.

One process owns: a camera source (read-only bridge child, or paced replay/synthetic),
an optional LiDAR/odometry source (read-only bridge child, separate environment), the
existing ``ShadowSession`` with one reused model worker (full-history policy, cached
preparation), a private append-only ``RunStore`` and a loopback-only display server.

Rendering is independent of inference: the camera display tap and the LiDAR reader run
on their own threads, the server reads snapshots, and the main loop never waits for
the model (one request in flight, bounded scheduling). Terminal commands are handled
between bounded steps. Nothing here can send anything to the robot: ``end`` and
``quit`` cancel *advisory* work only.
"""

from collections import OrderedDict, deque
from dataclasses import asdict
import json
import math
from pathlib import Path
from queue import Empty
import shutil
import threading
import time

import cv2

from .episodes import file_sha256
from .interactive_trials import (Trial, ReviewDialog, append_jsonl, as_instruction, contact_sheet,
                                 correction_record, expand_target, file_bytes, load_review,
                                 pixel_sha256, read_jsonl, utc_now)
from .interactive_guided import GuidedTest
from .shadow_display import target_fraction
from .shadow_sources import ObservationInbox
from .shadow_worker import MockWorker

HELP = """Commands (terminal; the browser only displays and accepts review clicks):
  target TEXT        new trial; expands to "Go to the TEXT and stop." (saved before inference)
  instruction TEXT   new trial with TEXT verbatim as the instruction
  mark [NOTE]        bind a mark to the prediction + camera frame the browser shows now
  note TEXT          verbatim note bound to the current trial and displayed IDs
  end                end the trial (advisory cancel only; no robot command) and review it
  review [ID|close]  list trials, open one (contact sheet + new append-only review), or close
  status             print a compact status line
  save               flush records and regenerate the contact sheet
  quit               end everything (no robot command is ever sent)
  help               this text
During a review dialog every line is an answer; `!cancel` discards it, `!quit` quits."""

GAP_S = 0.375  # the session's own native-history gap rule
SAFETY_QUESTION = ("Before the first trial: is the Go2 stationary, does the handheld remote own control "
                   "(no app/API control), is onboard avoidance on, and are body clearance and cable slack "
                   "checked? [yes/no]")
NEXT = {
    "ready": ("NEXT: in the browser page, type a target and press “Ask the model”. "
              "(This terminal is a log; `quit` closes the tool.)"),
    "started": ("NEXT: keep the robot still and wait ~2-3 s for the first answer (panel B); "
                "the trial stops by itself after ~{limit:.0f} s"),
    "answer": "NEXT: `mark <what you see>` to record whether the + is right · `end` to finish this trial",
    "marked": "NEXT: `mark <...>` again for a later answer · `end` to finish and review this trial",
    "ended": "NEXT: `review {id}` to answer the review questions · `target <object>` for a new trial · `quit`",
    "reviewed": "NEXT: `target <object>` for a new trial · `review` to list trials · `quit` to finish",
    "no_trial": "NEXT: `target <object>` to start a trial · `review` to list saved trials · `quit`",
}


def label_for(reason):
    if not reason:
        return "CURRENT"
    if reason.startswith("stale"):
        return "STALE"
    if reason.startswith(("malformed", "wrong", "invalid_inference", "unsolicited")):
        return "REJECTED"
    return "SUPERSEDED"


class PointingMockWorker(MockWorker):
    """MOCK for UI development only: alternates a canned grid target and OPOS-0.

    No model, no image content is used; the point and the curved path are fixed
    tokens chosen to exercise panels B/C. Every result keeps ``mock: True``.
    """

    def poll(self, now):
        response = super().poll(now)
        if response is None:
            return None
        result = response["result"]
        self.calls = getattr(self, "calls", 0) + 1
        if self.calls % 2:
            opos = 13 * 48 + 30  # grid cell (col 29, row 13) of the pinned 48x27 grid
            result["native_output"] = result["native_output"].replace("<opos_0>", f"<opos_{opos}>")
            result["pointing"] = {**result["pointing"], "opos": opos}
            result["waypoints"] = [[round(0.12 * (i + 1), 4), round(-0.015 * (i + 1) ** 1.5, 4),
                                    round(-0.05 * (i + 1), 4)] for i in range(10)]
        result["mock_notice"] = ("Canned alternating point/OPOS-0 and path; no network or RVQ inference, "
                                 "no semantic accuracy claim")
        return response


class TapInbox(ObservationInbox):
    """ObservationInbox that also keeps a small ring of display frames (not restamped).

    Called on the source thread, so panel A updates even while the main loop prepares
    a model request. Overflow of the session inbox is counted, never silent.
    """

    def __init__(self, capacity=16, ring=24, monotonic=time.monotonic):
        super().__init__(capacity)
        self.lock = threading.Lock()
        self.ring = OrderedDict()
        self.ring_size = ring
        self.monotonic = monotonic
        self.receipts = deque(maxlen=64)
        self.latest_id = None
        self.counts = {"offered": 0, "inbox_overflow": 0, "receipt_gaps": 0}
        self.last_gap = None
        self.jpeg = (None, None)

    def offer(self, observation, **kwargs):
        received = kwargs.get("received_monotonic_s")
        received = self.monotonic() if received is None else received
        kwargs["received_monotonic_s"] = received
        v = observation.visual
        if v is not None:
            meta = {
                "frame_id": kwargs["frame_id"], "source_id": kwargs["source_id"],
                "clock_epoch": observation.clock_epoch,
                "stamp_host": asdict(v.stamp_host),
                "stamp_robot": None if v.stamp_robot is None else asdict(v.stamp_robot),
                "available_s": kwargs["available_s"], "received_monotonic_s": received,
                "exposure_monotonic_s": kwargs.get("exposure_monotonic_s"),
                "monotonic_mapping": kwargs.get("monotonic_mapping"),
                "rectified": bool(v.rectified), "intrinsics_status": v.intrinsics_status,
                "distortion_model": v.distortion_model, "shape": list(v.image.shape),
            }
            with self.lock:
                if self.receipts and received - self.receipts[-1] > GAP_S:
                    self.counts["receipt_gaps"] += 1
                    self.last_gap = {"after_s": received - self.receipts[-1], "at_frame": meta["frame_id"]}
                self.receipts.append(received)
                self.ring[meta["frame_id"]] = (v.image, meta)
                while len(self.ring) > self.ring_size:
                    self.ring.popitem(last=False)
                self.latest_id = meta["frame_id"]
                self.counts["offered"] += 1
        ok = super().offer(observation, **kwargs)
        if not ok:
            with self.lock:
                self.counts["inbox_overflow"] += 1
        return ok

    def frame(self, frame_id):
        with self.lock:
            return self.ring.get(frame_id)

    def latest_jpeg(self, frame_id=None):
        with self.lock:
            frame_id = frame_id or self.latest_id
            item = self.ring.get(frame_id)
            cached_id, data = self.jpeg
        if item is None:
            return None, None
        if cached_id != frame_id:
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(item[0], cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, 85])
            data = buf.tobytes() if ok else None
            with self.lock:
                self.jpeg = (frame_id, data)
        return frame_id, data

    def status(self, now, silence_s):
        with self.lock:
            latest = None if self.latest_id is None else dict(self.ring[self.latest_id][1])
            receipts = list(self.receipts)
            counts = dict(self.counts)
            last_gap = self.last_gap
        recent = [t for t in receipts if t > now - 2.0]
        fps = None if len(recent) < 2 else (len(recent) - 1) / (recent[-1] - recent[0])
        out = {"latest": latest, "fps_receipt_2s": fps, "counts": counts, "last_gap": last_gap,
               "inbox_queued": self.queue.qsize(), "inbox_capacity": self.queue.maxsize}
        if latest is not None:
            age = now - latest["received_monotonic_s"]
            out["host_since_receipt_s"] = age
            out["since_estimated_exposure_s"] = (None if latest["exposure_monotonic_s"] is None
                                                 else now - latest["exposure_monotonic_s"])
            out["state"] = "INTERRUPTED" if age > silence_s else "RECEIVING"
        else:
            out["state"] = "WAITING FOR FIRST FRAME"
        return out


class PacedFeeder(threading.Thread):
    """Offers recorded/synthetic deliveries at their recorded pace; original stamps kept."""

    def __init__(self, deliveries, inbox, *, speed=1.0, monotonic=time.monotonic):
        super().__init__(daemon=True, name="paced-feeder")
        if not math.isfinite(speed) or not 0 < speed <= 20:
            raise ValueError("Replay speed 0..20 required")
        self.deliveries, self.inbox, self.speed, self.monotonic = deliveries, inbox, speed, monotonic
        self.stop_event = threading.Event()
        self.anchor = None
        self.ended = False
        self.error = None
        self.alive_flag = True

    def run(self):
        try:
            start = self.monotonic()
            for d in self.deliveries:
                if self.anchor is None:
                    self.anchor = (d.available_s, start, self.speed)
                due = start + (d.available_s - self.anchor[0]) / self.speed
                while not self.stop_event.is_set() and self.monotonic() < due:
                    self.stop_event.wait(min(0.01, max(0.0, due - self.monotonic())))
                if self.stop_event.is_set():
                    return
                self.inbox.offer(d.observation, source_id=d.source_id, frame_id=d.frame_id,
                                 available_s=d.available_s, provenance=d.provenance)
        except Exception as error:
            self.error = repr(error)
        finally:
            self.ended = True

    # Same small interface as LiveCameraSource.
    def get_nowait(self):
        return self.inbox.get_nowait()

    def drain_events(self):
        return []

    @property
    def alive(self):
        return not self.ended

    def close(self):
        self.stop_event.set()
        self.join(timeout=2)


class ResourceMonitor(threading.Thread):
    """Samples memory/disk every ``period_s``. RSS and MPS are reported separately, never summed."""

    def __init__(self, root, *, pids=None, limits, period_s=2.0, walk_every=5):
        super().__init__(daemon=True, name="resource-monitor")
        self.root, self.pids, self.limits = Path(root), pids or (lambda: {}), limits
        self.period_s, self.walk_every = period_s, walk_every
        self.lock = threading.Lock()
        self.value = {"low": [], "sampled": False}
        self.stop_event = threading.Event()
        self.history = deque(maxlen=600)
        self._n = 0

    def sample(self):
        import psutil
        own = psutil.Process()
        rss = {"app_parent": own.memory_info().rss}
        for name, pid in self.pids().items():
            if pid is None:
                continue
            try:
                rss[name] = psutil.Process(pid).memory_info().rss
            except psutil.Error:
                rss[name] = None
        vm = psutil.virtual_memory()
        disk = shutil.disk_usage(self.root)
        run_bytes = self.value.get("run_bytes")
        if run_bytes is None or self._n % self.walk_every == 0:
            run_bytes = file_bytes(self.root)
        low = []
        if vm.available < self.limits["min_available_memory_bytes"]:
            low.append(f"system available memory {vm.available / 2**30:.1f} GiB below limit")
        if rss["app_parent"] > self.limits["max_parent_rss_bytes"]:
            low.append(f"app RSS {rss['app_parent'] / 2**30:.2f} GiB above limit")
        if disk.free < self.limits["min_free_disk_bytes"]:
            low.append(f"free disk {disk.free / 2**30:.1f} GiB below limit")
        if run_bytes > self.limits["run_quota_bytes"]:
            low.append(f"run directory {run_bytes / 2**30:.2f} GiB above quota")
        value = {"sampled": True, "rss_bytes_by_process": rss, "system_available_bytes": vm.available,
                 "disk_free_bytes": disk.free, "run_bytes": run_bytes, "low": low,
                 "note": "per-process RSS; worker MPS memory is reported by the worker separately "
                         "and is not added to any RSS figure",
                 "sampled_host_monotonic_s": time.monotonic()}
        with self.lock:
            self.value = value
            self.history.append({k: value[k] for k in ("rss_bytes_by_process", "system_available_bytes",
                                                       "run_bytes", "sampled_host_monotonic_s")})
        return value

    def run(self):
        while not self.stop_event.is_set():
            try:
                self.sample()
            except Exception as error:
                with self.lock:
                    self.value = {**self.value, "error": repr(error)}
            self._n += 1
            self.stop_event.wait(self.period_s)

    def snapshot(self):
        with self.lock:
            return json.loads(json.dumps(self.value))

    def close(self):
        self.stop_event.set()
        self.join(timeout=3)


class InteractiveApp:
    def __init__(self, *, store, session, camera, tap, mode, lidar=None, lidar_recorder=None,
                 monitor=None, config, run_info, say=print, monotonic=time.monotonic):
        self.store, self.session, self.camera, self.tap = store, session, camera, tap
        self.mode, self.lidar, self.lidar_recorder = mode, lidar, lidar_recorder
        self.monitor, self.config, self.run_info = monitor, config, run_info
        self.say, self.monotonic = say, monotonic
        self.lock = threading.Lock()
        self.trial = None
        self.revision_trial = {}
        self.trials_started = 0
        self.trial_first_available = None
        self.trial_submitted = 0
        self.trial_frames = {}  # frame_id -> session observation meta (current trial only)
        self.feeding_paused = None
        self.predictions = OrderedDict()  # prediction_id -> public record (bounded)
        self.pred_paths = {}
        self.last_receipt = None
        self.review = None
        self.dialog = None
        self.safety_confirmation = None  # one combined operator answer per run
        self.pending_spec = None
        self.single_answer = False
        self.guided = GuidedTest(self)
        self.viewer = {}
        self.resource_stop = None
        self.camera_down = False
        self.model_identity = None
        self.public = {}
        self.last_public = -math.inf
        self.quit = False
        self.termination = None
        self.last_inference_s = None

    # ---------- browser/server side (thread-safe getters) ----------
    def report_view(self, payload, now):
        with self.lock:
            self.viewer = {"prediction_id": payload.get("prediction_id"),
                           "camera_frame_id": payload.get("camera_frame_id"),
                           "lidar_view": payload.get("lidar_view"),
                           "reported_host_monotonic_s": now}

    def viewer_snapshot(self):
        with self.lock:
            return dict(self.viewer)

    def prediction_image_path(self, prediction_id):
        with self.lock:
            rel = self.pred_paths.get(prediction_id)
        if rel is None:
            return None
        path = (self.store.root / rel).resolve()
        if self.store.root.resolve() not in path.parents or not path.is_file():
            return None
        return path

    def state(self, now):
        with self.lock:
            public = dict(self.public)
            viewer = dict(self.viewer)
        silence = self.config["camera_silence_s"]
        public["camera"] = None if self.tap is None else self.tap.status(now, silence)
        public["lidar"] = None if self.lidar is None else self.lidar.snapshot(now)
        public["resources"] = None if self.monitor is None else self.monitor.snapshot()
        public["viewer"] = viewer
        public["host_monotonic_s"] = now
        return public

    # ---------- trials ----------
    def _refuse_trial(self):
        if self.resource_stop:
            return "Resources are low (" + "; ".join(self.resource_stop) + "); new trials are stopped."
        if self.dialog is not None:
            return "Finish or `!cancel` the review dialog first."
        if self.session is None:
            return "REVIEW-only mode: no camera/model; trials cannot start."
        if not self.session.worker.ready:
            return "Model is not ready yet (loading); try again shortly."
        if self.trials_started >= self.config["max_trials"]:
            return f"Trial budget reached ({self.config['max_trials']}); restart the tool for more."
        return None

    def start_trial(self, spec, now, single_answer=False):
        refusal = self._refuse_trial()
        if refusal:
            self.say(refusal)
            return None
        if self.trial is not None:
            self.end_trial("superseded_by_new_trial", now, review=False)
        self.close_review(quiet=True)
        context = {
            "source_mode": self.session.mode,
            "temporal_context": {
                "policy": "existing full-history: reconstruct the complete causal instruction "
                          "history per request (latest-anchored 4 Hz grid + native SlowFast)",
                "bounded_context_policy_enabled": False,
                "preparation_mode": self.session.preparation_mode,
                "limits": asdict(self.session.limits),
                "warmup_s": self.config["warmup_s"],
                "max_requests_per_trial": self.config["max_requests_per_trial"],
                "trial_max_s": self.config["trial_max_s"],
            },
            "operator_safety_confirmation": self.safety_confirmation,
            "single_answer_guided_test": single_answer,
            "session_id": self.session.session_id,
            "instruction_revision": self.session.revision + 1,
            "model_identity_file": self.run_info.get("model_identity_file"),
        }
        trial = Trial.start(self.store, spec, now, **context)
        self.session.instruction_changed(spec["instruction"], now)
        assert self.session.revision == context["instruction_revision"]
        self.revision_trial[self.session.revision] = trial.id
        self.trial = trial
        self.trials_started += 1
        self.trial_first_available = None
        self.trial_submitted = 0
        self.trial_frames = {}
        self.feeding_paused = None
        self.single_answer = single_answer
        if self.lidar_recorder is not None:
            self.lidar_recorder.start(trial.id)
        self.store.event("trial_started", now, trial_id=trial.id, instruction=spec["instruction"],
                         template_id=spec["template_id"])
        self.say(f"Trial {trial.id} started. Instruction: {spec['instruction']!r}"
                 + (f" (template {spec['template_id']})" if spec["template_id"] else ""))
        if not single_answer:
            self.say(NEXT["started"].format(limit=self.session.limits.history_s))
        return trial

    def request_trial(self, spec, now):
        """Start a trial; the first one in a run asks the single safety question first."""
        if self.safety_confirmation is not None:
            return self.start_trial(spec, now)
        refusal = self._refuse_trial()
        if refusal:
            self.say(refusal)
            return None
        self.pending_spec = spec
        self.say(SAFETY_QUESTION)
        return None

    def _answer_safety(self, line, now):
        answer = line.strip().lower()
        if answer not in ("yes", "y", "no", "n"):
            self.say("Please answer yes or no.\n" + SAFETY_QUESTION)
            return
        spec, self.pending_spec = self.pending_spec, None
        if answer in ("no", "n"):
            self.store.event("safety_question_declined", now)
            self.say("Not started. Nothing was sent to the robot.\n" + NEXT["no_trial"])
            return
        self.safety_confirmation = {"answer": "yes", "question": SAFETY_QUESTION, "wall_utc": utc_now(),
                                    "note": "operator confirmation; the app cannot verify it and sends nothing"}
        self.store.event("safety_question_confirmed", now, **self.safety_confirmation)
        self.start_trial(spec, now)

    def end_trial(self, reason, now, *, review=True):
        trial = self.trial
        if trial is None:
            self.say("No active trial.")
            return None
        # Advisory cancel only: invalidates the in-flight request; nothing goes to the robot.
        self.session.cancel(now)
        if self.lidar_recorder is not None:
            if self.config.get("record_lidar_continuously"):
                self.lidar_recorder.start("between_trials")  # odometry across manual viewpoint changes
            else:
                self.lidar_recorder.stop()
        trial.end(reason, now, predictions=list(trial.prediction_ids), marks=trial.marks,
                  notes=trial.notes, inflight_at_end=self.session.inflight is not None,
                  history_error=self.session.history_error,
                  frames_received=len(self.trial_frames))
        self.trial = None
        single, self.single_answer = self.single_answer, False
        self.store.event("trial_ended", now, trial_id=trial.id, reason=reason)
        self.say(f"Trial {trial.id} ended ({reason}). Advisory cancelled; no robot command was sent.")
        if single:
            contact_sheet(self.store, trial.id, self._prediction_array)  # file only; page stays on the test
        else:
            self.open_review(trial.id, now, dialog=review)
        if not review and not single:
            # Automatic ends never open the question dialog: the next typed line stays a command.
            self.say(NEXT["ended"].format(id=trial.id))
        return trial

    # ---------- review ----------
    def open_review(self, trial_id, now, *, dialog=True):
        try:
            data = load_review(self.store, trial_id)
        except (FileNotFoundError, ValueError) as error:
            self.say(str(error))
            return
        for p in data["predictions"]:
            if p.get("source_image") and p["source_image"].get("path"):
                with self.lock:
                    self.pred_paths[p["prediction_id"]] = p["source_image"]["path"]
        path = contact_sheet(self.store, trial_id, self._prediction_array)
        data = load_review(self.store, trial_id)
        with self.lock:
            self.review = {"trial_id": trial_id, "data": data}
        self.say(f"REVIEW {trial_id}: contact sheet {path} (also in the browser).")
        if dialog:
            self.dialog = ReviewDialog(self.store, trial_id, reviewer_default=self.config.get("reviewer"),
                                       now=now)
            self.say(self.dialog.prompt())
        self._publish(now, force=True)

    def close_review(self, quiet=False):
        with self.lock:
            had = self.review is not None
            self.review = None
        if had and not quiet:
            self.say("Review closed; showing live panels only.")

    def _refresh_review(self):
        with self.lock:
            review = self.review
        if review is not None:
            data = load_review(self.store, review["trial_id"])
            with self.lock:
                if self.review is not None and self.review["trial_id"] == review["trial_id"]:
                    self.review = {"trial_id": review["trial_id"], "data": data}

    def mark_image(self, trial_id, mark_id):
        for mark in read_jsonl(self.store.trial_dir(trial_id) / "marks.jsonl"):
            frame = mark.get("camera_frame") or {}
            if mark["mark_id"] == mark_id and frame.get("image"):
                return self.store.load_image(frame["image"])
        return None

    def _prediction_array(self, pred):
        ref = pred.get("source_image") or {}
        if not ref.get("path"):
            return None
        bgr = cv2.imread(str(self.store.root / ref["path"]), cv2.IMREAD_COLOR)
        return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def correction(self, payload, now):
        """Browser click in REVIEW mode; queued to the main thread via this lock-free call."""
        with self.lock:
            review = self.review
        if review is None:
            raise ValueError("Corrections are accepted only while a trial is open in REVIEW")
        preds = {p["prediction_id"]: p for p in review["data"]["predictions"]}
        pred = preds.get(payload.get("prediction_id"))
        if pred is None:
            raise ValueError("Unknown prediction for this trial")
        record = correction_record(self.store, review["trial_id"], pred, payload, now,
                                   self.config.get("reviewer"))
        append_jsonl(self.store.trial_dir(review["trial_id"]) / "corrections.jsonl", record)
        self.store.event("target_correction_saved", now, trial_id=review["trial_id"],
                         correction_id=record["correction_id"], prediction_id=pred["prediction_id"])
        self._refresh_review()
        self.say(f"Correction {record['correction_id']} saved for {pred['prediction_id']} "
                 f"at ({record['x_fraction']:.3f}, {record['y_fraction']:.3f}) of its own input image "
                 "(stored separately; not a model input).")
        return record

    # ---------- marks and notes ----------
    def _bound_view(self, typed_s, viewer):
        current = self.public.get("prediction") or {}
        fresh = viewer.get("reported_host_monotonic_s") is not None and typed_s - viewer["reported_host_monotonic_s"] <= 2.0
        pred_id = viewer.get("prediction_id") if fresh else current.get("prediction_id")
        frame_id = viewer.get("camera_frame_id") if fresh else (self.tap.latest_id if self.tap else None)
        binding = "browser_reported_view" if fresh else "server_latest_browser_view_not_reported_recently"
        return pred_id, frame_id, binding

    def _frame_record(self, frame_id):
        if frame_id is None or self.tap is None:
            return None
        item = self.tap.frame(frame_id)
        if item is None:
            return {"frame_id": frame_id, "image": None, "reason": "frame no longer in display ring"}
        image, meta = item
        session_meta = self.trial_frames.get(frame_id)
        if session_meta is not None:
            ref = {"store": "session_observation",
                   "path": str(Path(session_meta["observation_path"]).resolve().relative_to(self.store.root.resolve())),
                   "file_sha256": session_meta["observation_sha256"], "pixel_sha256": pixel_sha256(image),
                   "shape": list(image.shape), "channel_order": "RGB", "encoding": "npz_compressed_lossless"}
        else:
            ref = self.store.store_image(image)
        return {**meta, "image": ref}

    def mark(self, text, typed_s, viewer, now):
        if self.trial is None:
            self.say("`mark` needs an active trial (use `note` in REVIEW).")
            return None
        pred_id, frame_id, binding = self._bound_view(typed_s, viewer)
        pred = self.predictions.get(pred_id)
        shown = self.public.get("prediction") or {}
        self.trial.marks += 1
        record = {
            "mark_id": f"M{self.trial.marks:03d}", "trial_id": self.trial.id,
            "verbatim_note": text, "typed_host_monotonic_s": typed_s, "typed_wall_utc": utc_now(),
            "binding": binding, "viewer_report": viewer,
            "prediction": None if pred is None else {
                k: pred[k] for k in ("prediction_id", "trial_id", "request_sequence", "label_at_receipt",
                                     "source_frame", "source_image", "target_fraction", "opos_not_visible")},
            "prediction_label_when_marked": shown.get("label") if shown.get("prediction_id") == pred_id else None,
            "prediction_from_other_trial": pred is not None and pred["trial_id"] != self.trial.id,
            "camera_frame": self._frame_record(frame_id),
            "source": "human_operator", "model_feedback": False,
        }
        self.trial.append("marks.jsonl", record)
        self.store.event("mark_saved", now, trial_id=self.trial.id, mark_id=record["mark_id"],
                         prediction_id=pred_id, frame_id=frame_id)
        self.say(f"Mark {record['mark_id']} → prediction {pred_id or 'none shown'}, camera frame {frame_id} ({binding}).")
        self.say(NEXT["marked"])
        return record

    def note(self, text, typed_s, viewer, now):
        if not text.strip():
            self.say("`note` needs text.")
            return None
        with self.lock:
            review = self.review
        trial_id = self.trial.id if self.trial is not None else (review["trial_id"] if review else None)
        pred_id, frame_id, binding = self._bound_view(typed_s, viewer)
        record = {"note_id": None, "trial_id": trial_id, "verbatim_note": text,
                  "typed_host_monotonic_s": typed_s, "typed_wall_utc": utc_now(),
                  "context": "live_trial" if self.trial is not None else "review" if review else "no_trial",
                  "prediction_id": pred_id, "camera_frame_id": frame_id, "binding": binding,
                  "camera_frame": self._frame_record(frame_id) if self.trial is not None else None,
                  "source": "human_operator", "model_feedback": False}
        if trial_id is None:
            record["note_id"] = "N-run"
            append_jsonl(self.store.root / "notes.jsonl", record)
        else:
            folder = self.store.trial_dir(trial_id)
            record["note_id"] = f"N{len(read_jsonl(folder / 'notes.jsonl')) + 1:03d}"
            append_jsonl(folder / "notes.jsonl", record)
            if self.trial is not None:
                self.trial.notes += 1
        self.store.event("note_saved", now, trial_id=trial_id, note_id=record["note_id"])
        self._refresh_review()
        self.say(f"Note {record['note_id']} saved verbatim for {trial_id or 'the run'}.")
        return record

    # ---------- commands ----------
    def handle(self, line, typed_s, viewer, now):
        if self.pending_spec is not None:
            if line.strip() == "!quit":
                self.pending_spec = None
                return self.handle("quit", typed_s, viewer, now)
            self._answer_safety(line, now)
            return True
        if self.dialog is not None:
            if line.strip() == "!quit":
                self.dialog = None
                return self.handle("quit", typed_s, viewer, now)
            message, _ = self.dialog.feed(line, now)
            self.say(message)
            if self.dialog.done:
                self.dialog = None
                self._refresh_review()
                self.say(NEXT["reviewed"])
            return True
        command, _, rest = line.lstrip().partition(" ")  # rest stays verbatim (notes)
        command = command.lower()
        try:
            if command == "target":
                self.request_trial(expand_target(rest), now)
            elif command == "instruction":
                self.request_trial(as_instruction(rest), now)
            elif command == "mark":
                self.mark(rest, typed_s, viewer, now)
            elif command == "note":
                self.note(rest, typed_s, viewer, now)
            elif command == "end":
                self.end_trial("operator_end", now)
            elif command == "review":
                arg = rest.strip()
                if arg == "close":
                    self.close_review()
                elif arg:
                    self.open_review(arg, now)
                else:
                    trials = self.store.trials()
                    if not trials:
                        self.say("No trials yet.")
                    for tid, row in trials.items():
                        labels = len(read_jsonl(self.store.root / "trials" / tid / "human_labels.jsonl"))
                        self.say(f"{tid}  {row.get('kind')}  {row.get('instruction')!r}  "
                                 f"end={row.get('end_reason', 'ACTIVE')}  reviews={labels}")
            elif command == "status":
                self.say(json.dumps(self.status_line(now)))
            elif command == "save":
                self.save(now)
            elif command in ("quit", "exit"):
                self.quit, self.termination = True, "operator_quit"
                return False
            elif command in ("help", "?"):
                self.say(HELP)
            elif command:
                self.say("Unknown command. " + HELP)
        except ValueError as error:
            self.say(f"Not done: {error}")
        return True

    def save(self, now):
        target = self.trial.id if self.trial else (self.review or {}).get("trial_id")
        if target:
            path = contact_sheet(self.store, target, self._prediction_array)
            self.say(f"Saved. Contact sheet: {path}")
        status = self.status_line(now)
        tmp = self.store.root / "run_status.tmp.json"
        tmp.write_text(json.dumps(status, indent=2) + "\n")
        tmp.replace(self.store.root / "run_status.json")
        self.store.event("save", now, trial_id=target)
        self.say(f"Records are appended as they happen; status written to {self.store.root / 'run_status.json'}.")

    def status_line(self, now):
        s = self.session
        pred = self.public.get("prediction") or {}
        return {
            "mode": self.mode, "trial": None if self.trial is None else self.trial.id,
            "model": self.public.get("model"),
            "prediction": pred.get("prediction_id"), "prediction_label": pred.get("label"),
            "session_counts": None if s is None else s.counts,
            "history_error": None if s is None else s.history_error,
            "resource_stop": self.resource_stop,
            "robot_commands_sent": 0,
        }

    # ---------- main loop ----------
    def _model_state(self):
        s = self.session
        if s is None:
            return {"state": "NOT LOADED (review only)"}
        w = s.worker
        state = "LOADING" if not w.ready and not s.closed else "INFERRING" if s.inflight else "READY"
        if s.closed:
            state = "CLOSED"
        return {"state": state, "backend": self.run_info.get("backend"), "in_flight": s.inflight is not None,
                "pending": bool(s.pending), "last_inference_s": self.last_inference_s,
                "identity": self.model_identity, "preparation_mode": s.preparation_mode,
                "history_frames": len(s.history), "history_error": s.history_error}

    def _record_prediction(self, now):
        request, record = self.session.displayed
        if self.last_receipt == request.folder:
            return
        self.last_receipt = request.folder
        identity = request.context["identity"]
        trial_id = self.revision_trial.get(identity["instruction_revision"])
        response = record["response"]
        result = response.get("result", {}) if isinstance(response.get("result"), dict) else {}
        job = request.job
        image = None
        indices = result.get("native_selected_frame_indices")
        if isinstance(indices, list) and indices and type(indices[-1]) is int and 0 <= indices[-1] < len(job["frames"]):
            i = indices[-1]
            path = Path(job["frames"][i]["path"]) if result.get("mock") else request.folder / "native" / f"input_{i:04d}.png"
            if path.is_file():
                image = {"path": str(path.resolve().relative_to(self.store.root.resolve())),
                         "file_sha256": file_sha256(path), "job_frame_index": i,
                         "kind": "mock_prepared_rectified_frame" if result.get("mock")
                         else "native_model_space_input (rectified, official processor output)"}
        source = request.context["current_source"]
        pointing = result.get("pointing") if isinstance(result.get("pointing"), dict) else {}
        semantics = record.get("semantics") or {}
        timing = result.get("timing") if isinstance(result.get("timing"), dict) else {}
        pred = {
            "prediction_id": f"{trial_id or 'untracked'}-P{identity['sequence']:03d}",
            "trial_id": trial_id, "request_sequence": identity["sequence"],
            "instruction_revision": identity["instruction_revision"],
            "request_directory": str(request.folder.relative_to(self.store.root)),
            "instruction": job["instruction"],
            "received_host_monotonic_s": record["receipt_monotonic_s"], "received_wall_utc": utc_now(),
            "rejection_reason_at_receipt": record["rejection_reason"],
            "label_at_receipt": label_for(record["rejection_reason"]),
            "backend": response.get("backend"), "mock": bool(result.get("mock")),
            "native_output": result.get("native_output"),
            "pointing": pointing, "target_fraction": target_fraction(pointing),
            "opos_not_visible": pointing.get("opos") == 0,
            "waypoints": result.get("waypoints"),
            "waypoint_convention": result.get("waypoint_convention"),
            "waypoint_time_base": None,
            "native_stop": result.get("decoder_stop"), "explicit_stop_code": result.get("explicit_stop_code"),
            "model_completion_claim": semantics.get("model_completion_claim"),
            "completion_verified": False, "execution_status": "NOT EXECUTED — advisory only; no command path",
            "source_image": image,
            "source_frame": {"frame_id": source["frame_id"], "observation_s": source["observation_s"],
                             "available_s": source["available_s"], "clock": source["stamp"]["clock"],
                             "semantics": source["stamp"]["semantics"],
                             "uncertainty_s": source["stamp"]["uncertainty_s"],
                             "clock_epoch": source["clock_epoch"],
                             "received_monotonic_s": source.get("received_monotonic_s"),
                             "exposure_monotonic_s": source.get("exposure_monotonic_s")},
            "history_frames_in_job": len(job["frames"]),
            "display_timing_at_receipt": record["display_timing"],
            "inference_duration_s": response.get("inference_duration_s"),
            "native_generate_s": timing.get("native_generate_s"),
            "worker_memory": result.get("memory"),
            "model_output_not_a_label": True,
        }
        self.last_inference_s = response.get("inference_duration_s")
        with self.lock:
            self.predictions[pred["prediction_id"]] = pred
            while len(self.predictions) > 256:
                self.predictions.popitem(last=False)
            if image:
                self.pred_paths[pred["prediction_id"]] = image["path"]
        if trial_id is not None:
            folder = self.store.root / "trials" / trial_id
            append_jsonl(folder / "predictions.jsonl", pred)
            if self.trial is not None and self.trial.id == trial_id:
                self.trial.prediction_ids.append(pred["prediction_id"])
                what = ("model says the target is NOT VISIBLE (OPOS-0)" if pred["opos_not_visible"]
                        else "model points at the + in panel B" if pred["target_fraction"] else "no target point")
                self.say(f"Answer {pred['prediction_id'].rsplit('-', 1)[-1]} after "
                         f"{(pred['inference_duration_s'] or 0):.1f} s ({pred['label_at_receipt']}): {what}.")
                if not self.single_answer:
                    self.say(NEXT["answer"])
        self.store.event("prediction_received", now, prediction_id=pred["prediction_id"],
                         trial_id=trial_id, label=pred["label_at_receipt"],
                         reason=record["rejection_reason"])
        if trial_id is not None and (self.trial is None or self.trial.id != trial_id):
            with self.lock:
                reviewing = self.review is not None and self.review["trial_id"] == trial_id
            if reviewing:
                contact_sheet(self.store, trial_id, self._prediction_array)
            self._refresh_review()
            self.say(f"Late result {pred['prediction_id']} recorded with its original trial "
                     f"({pred['label_at_receipt']}: {record['rejection_reason']}).")

    def _current_prediction(self, now):
        if self.session is None or not self.session.displayed:
            return None
        request, record = self.session.displayed
        pid = f"{self.revision_trial.get(request.context['identity']['instruction_revision'], 'untracked')}-P{request.context['identity']['sequence']:03d}"
        pred = self.predictions.get(pid)
        if pred is None:
            return None
        reason = record["rejection_reason"] or self.session._rejection(request, now)
        return {**pred, "label": label_for(reason), "current_rejection_reason": reason,
                "display_timing": self.session.timing(request, now),
                "image_url": f"/api/pred_image?id={pid}" if pred["source_image"] else None,
                "active_trial": self.trial.id if self.trial else None}

    def _publish(self, now, force=False):
        if not force and now - self.last_public < 0.2:
            return
        with self.lock:
            review = self.review
        public = {
            "mode": self.mode,
            "banner": "ADVISORY ONLY — NO ROBOT COMMANDS",
            "view": "REVIEW" if review else self.mode,
            "run_directory": str(self.store.root),
            "trial": None if self.trial is None else {
                "trial_id": self.trial.id, "instruction": self.trial.record["instruction"],
                "operator_text": self.trial.record["operator_text"], "kind": self.trial.record["kind"],
                "template_id": self.trial.record["template_id"],
                "elapsed_s": now - self.trial.record["started_host_monotonic_s"],
                "feeding_paused": self.feeding_paused,
                "requests": self.trial_submitted},
            "model": self._model_state(),
            "prediction": self._current_prediction(now),
            "guided": self.guided.public(now),
            "review": None if review is None else {
                "trial_id": review["trial_id"], **review["data"],
                "dialog_open": self.dialog is not None},
            "recording": {"trial": None if self.trial is None else self.trial.id,
                          "lidar": None if self.lidar_recorder is None else self.lidar_recorder.status(),
                          "session_observations": None if self.session is None else self.session.observation_count},
            "resource_stop": self.resource_stop,
            "clocks": self.run_info.get("clocks"),
            "robot_commands_sent": 0,
        }
        with self.lock:
            self.public = public
        self.last_public = now

    def step(self, now):
        """One bounded iteration. Never blocks on the model."""
        s = self.session
        if s is None:
            self._publish(now)
            return
        # 1. Camera deliveries -> session (only during an active, unpaused trial).
        for _ in range(8):
            try:
                delivery = self.camera.get_nowait()
            except Empty:
                break
            if self.trial is None or self.feeding_paused:
                continue
            received = self.monotonic()
            delivery.provenance["session_receive_monotonic_s"] = received
            s.receive(delivery, received)
            if s.latest and s.latest["frame_id"] == delivery.frame_id:
                self.trial_frames[delivery.frame_id] = s.latest
                if self.trial_first_available is None:
                    self.trial_first_available = delivery.available_s
        now = self.monotonic()
        for item in self.camera.drain_events():
            s.event(item.pop("event"), now, **item)
        if self.lidar is not None:
            for item in self.lidar.drain_events():
                self.store.event(item.pop("event"), now, **item)
        camera = self.tap.status(now, self.config["camera_silence_s"])
        down = camera["state"] == "INTERRUPTED"
        if down and not self.camera_down and self.trial is not None:
            s.event("camera_interrupted", now, silence_s=camera.get("host_since_receipt_s"))
            s.invalidate_history("camera_interrupted", now)  # report, never re-feed old frames
        if not down and self.camera_down:
            s.event("camera_resumed", now)
        self.camera_down = down
        if self.trial is not None and s.history_error and not self.feeding_paused:
            # Inference is paused; stop archiving frames for it (bounded disk).
            self.feeding_paused = s.history_error
            self.say(f"Inference paused for {self.trial.id}: {s.history_error}. The trial ends after its last "
                     "result; start a new target to continue.")
        # 2. Responses first, so a result is displayed before the next job is prepared.
        s.tick(now, allow_submit=False)
        if s.displayed:
            self._record_prediction(now)
        self._publish(self.monotonic(), force=True)
        # 3. Bounded scheduling: at most one new request, only for the active trial.
        now = self.monotonic()
        if self.trial is not None:
            if self.trial_submitted >= (1 if self.single_answer else self.config["max_requests_per_trial"]):
                s.pending = False
                if self.single_answer and not self.feeding_paused:
                    # One answer only: stop archiving frames while the operator decides.
                    self.feeding_paused = "single_answer_requested"
            warm = (self.trial_first_available is not None and s.latest is not None
                    and s.latest["available_s"] - self.trial_first_available >= self.config["warmup_s"])
            before = s.counts["submitted"]
            s.tick(now, allow_submit=warm)
            self.trial_submitted += s.counts["submitted"] - before
            if self.single_answer:
                pass  # the guided test ends on Save/Cancel (or its own timeout), never mid-decision
            elif now - self.trial.record["started_host_monotonic_s"] > self.config["trial_max_s"]:
                self.end_trial("trial_time_limit", now, review=False)
            elif self.feeding_paused and s.inflight is None:
                # No further inference is possible in this trial (history limit, gap or
                # interruption); end it once its last in-flight result has been recorded.
                self.end_trial("inference_paused: " + self.feeding_paused, now, review=False)
        # 4. Resources: stop new trials and finalize without deleting anything.
        if self.monitor is not None:
            low = self.monitor.snapshot().get("low") or []
            if low and not self.resource_stop:
                self.resource_stop = low
                self.store.event("resource_limit", now, reasons=low)
                self.say("RESOURCE LIMIT: " + "; ".join(low) + ". New trials stopped; recordings kept.")
                if self.trial is not None:
                    self.end_trial("resource_limit", now, review=False)
        # 5. Browser test actions and reconciliation with whatever ended or replaced a trial above.
        self.guided.process(self.monotonic())
        if self.model_identity is None and s.worker.ready:
            path = getattr(s.worker, "output", None)
            model = None if path is None else Path(path) / "model.json"
            self.model_identity = (json.loads(model.read_text()) if model and model.exists()
                                   else {"backend": self.run_info.get("backend"), "note": "mock: no model"})
            self.model_identity = {k: self.model_identity.get(k) for k in
                                   ("checkpoint", "upstream_code", "patch_sha256", "runtime", "load_s",
                                    "load_memory", "preparation_mode", "tensor_evidence", "backend", "note")}
            self.store.event("model_ready", now, identity=self.model_identity)
            self.say("Model ready.\n" + NEXT["ready"])
        self._publish(self.monotonic())

    def close(self, now):
        if self.dialog is not None:
            self.store.event("review_discarded", now, trial_id=self.dialog.trial_id, reason="quit_during_dialog")
            self.dialog = None
        if self.trial is not None:
            self.end_trial(self.termination or "shutdown", now, review=False)
