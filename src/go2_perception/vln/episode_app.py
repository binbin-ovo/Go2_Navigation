"""Browser and archive integration for the opt-in episode. One trial owns its entire history."""

from copy import deepcopy
import json
from queue import Empty, Full
import threading


from .episode_grounding import ground, object_reference, reference_matches_target, valid_roi
from .episode_mock import EpisodeMockWorker
from .episode_page import episode_page
from .five_move import SCHEMA
from .interactive_app import InteractiveApp
from go2_perception.navigation import target_spec
from .shadow import write_json
from .supervised import Hold, digest
from .bound_proposal import BoundProposalUI


class EpisodeUI(BoundProposalUI):
    """Share archived image and proposal validation with the one-action application."""

    def __init__(self, app, cap, executor, live, nonce, settings, fixture):
        super().__init__(app, cap, live, executor)
        self.nonce, self.settings, self.fixture = nonce, settings, fixture
        self.episode = dict(
            schema=SCHEMA,
            episode_id=nonce,
            state="IDLE",
            moves=0,
            attempts=[],
            outcome=None,
            target_found=False,
            cleanup_unverified=False,
            max_moves=settings["max_moves"],
            mode="SEARCH",
            reason="awaiting_area_authorization",
        )
        self.grant = self.ref = self.rejected = self.uncertain_review = None
        self.processed_sequence = 0
        self.waiting = False
        self.abort = threading.Event()
        self.last_error = None
        self.cleanup_uncertain = False
        self.command_sends = 0
        self.quit_requested = False

    def enqueue(self, payload, now):
        if payload.get("action") in ("stop", "cancel", "quit"):
            self.abort.set()
            self.executor.cancel()  # direct from HTTP/terminal, independent of the main event loop
            if payload["action"] == "quit":
                self.quit_requested = True
            return
        try:
            self.queue.put_nowait((dict(payload), now))
        except Full:
            raise Hold("episode_ui_queue_full")

    def send(self, kind, **fields):
        self.executor.send(dict(type=kind, episode_id=self.nonce, **fields))

    def authorize(self, p, queued):
        if self.grant or self.abort.is_set() or self.episode["state"] != "IDLE":
            raise Hold("episode_already_armed_or_closed")
        if p.get("area_inspected") is not True or p.get("exclusive_and_intervention_confirmed") is not True:
            raise Hold("area_and_independent_supervision_must_be_confirmed")
        spec = target_spec(p["target"])
        target, operator = spec["operator_text"], str(p["operator"]).strip()
        if not 1 <= len(operator) <= 100:
            raise Hold("named_operator_required")
        if self.settings.get("reference_path"):
            if not reference_matches_target(self.settings["reference_target"], target):
                raise Hold(f"reference_names_another_target: photo named {self.settings['reference_target']!r}, "
                           f"requested {target!r}. Identify this description on an image or supply a photo named for it.")
            self.ref = object_reference(
                self.settings["reference_path"], self.settings["reference_roi"], target, operator
            )
        guard = (self.executor.latest_guards or {}).get("snapshot", {})
        latest = self.app.tap.frame(self.app.tap.latest_id)
        now = self.app.monotonic()
        if (
            not latest
            or not guard
            or now - queued > 1.0
            or not 0 <= now - guard["pose"]["monotonic_s"] <= self.cap["limits"]["state_age_s"]
        ):
            raise Hold("current_camera_guard_and_fresh_authorization_required")
        epoch = latest[1]["clock_epoch"]
        if self.live and (
            guard.get("capture_clock_epoch") != epoch or not guard.get("capture_bound_basis")
        ):
            raise Hold("camera_epoch_or_bound_unverified")
        self.app.safety_confirmation = dict(
            operator=operator, area_inspected=True, exclusive_and_intervention_confirmed=True
        )
        while not self.app.tap.queue.empty():
            self.app.tap.queue.get_nowait()
        trial = self.app.start_trial(spec, now, single_answer=False)
        if trial is None:
            raise Hold("model_or_source_not_ready")
        self.instruction_id = f"{self.nonce}.r{self.app.session.revision}"
        self.grant = dict(
            schema=SCHEMA,
            episode_id=self.nonce,
            session_id=self.app.session.session_id,
            trial_id=trial.id,
            instruction_id=self.instruction_id,
            instruction=spec["instruction"],
            target=target,
            operator=operator,
            clock_epoch=epoch,
            origin=guard["pose"],
            capability_sha256=digest(self.cap),
            issued=now,
            deadline=now + self.settings["seconds"],
            area_inspected=True,
            exclusive_and_intervention_confirmed=True,
            **{k: self.settings[k] for k in ("max_moves", "area_radius_m", "alignment_deg")},
        )
        if self.ref:
            self.app.store.event(
                "identified_reference",
                now,
                target=target,
                operator=operator,
                source_sha256=self.ref["sha256"],
                roi=self.ref["roi"],
            )
        self.waiting = True
        self.send("episode_arm", grant=self.grant)

    def review_image(self, payload):
        e = self.episode
        last = e.get("last_observation") or {}
        if (
            not self.grant
            or payload.get("operator") != self.grant["operator"]
            or not last
            or (payload.get("prediction_id"), payload.get("image_sha256"))
            != (last["prediction_id"], last["image_sha256"])
        ):
            raise Hold("review_requires_same_displayed_image_and_operator")
        binding = {k: payload[k] for k in ("operator", "prediction_id", "image_sha256")}
        if payload["action"] == "goal":
            self.send("episode_goal", **binding)
            return
        if e["state"] != "IDENTIFYING":
            raise Hold("identification_not_pending")
        pred = self.app.predictions[last["prediction_id"]]
        if payload["action"] == "uncertain":
            self.uncertain_review = binding
            self.app.store.event(
                "episode_instance_review",
                self.app.monotonic(),
                **binding,
                target=self.grant["target"],
                decision="uncertain",
            )
            self.waiting = True
            self.send("episode_identified", **binding)
            return
        point = pred["target_fraction"]
        roi = payload.get("roi")
        if not valid_roi(roi):
            raise Hold("Draw a rectangle around the marked object in panel B before identifying or rejecting it.")
        if point is None or not (roi[0] <= point[0] <= roi[2] and roi[1] <= point[1] <= roi[3]):
            raise Hold("mark_must_be_inside_named_object_roi")
        path = self.app.prediction_image_path(pred["prediction_id"])
        if path is None:
            raise Hold("review_image_unavailable")
        ref = object_reference(path, roi, self.grant["target"], payload["operator"])
        if ref["sha256"] != last["image_sha256"]:
            raise Hold("review_image_changed")
        if payload["action"] == "identify":
            self.ref = ref
        else:
            self.rejected = ref
            self.uncertain_review = binding
        self.app.store.event(
            "episode_instance_review",
            self.app.monotonic(),
            **binding,
            target=self.grant["target"],
            decision=payload["action"],
            roi=roi,
        )
        self.waiting = True
        self.send("episode_identified", **binding)

    def submit_prediction(self, pred):
        self.processed_sequence = pred["request_sequence"]
        if pred["rejection_reason_at_receipt"]:
            self.app.store.event(
                "episode_stale_result",
                self.app.monotonic(),
                prediction_id=pred["prediction_id"],
                reason=pred["rejection_reason_at_receipt"],
            )
            return
        proposal, path = self.proposal(pred)
        context = json.loads((self.app.store.root / pred["request_directory"] / "snapshot.json").read_text())
        if context["history"][-1]["source"]["frame_id"] != pred["source_frame"]["frame_id"]:
            raise Hold("current_image_history_mismatch")
        evidence = ground(
            pred, path, context["history"][-1]["rectification"], self.grant["target"], self.ref, self.rejected
        )
        if (
            not self.live
            and self.grant["target"] == self.fixture["target"]
            and pred["mock"]
            and evidence["status"] == "needs_identification"
        ):
            x, y = pred["target_fraction"]
            evidence.update(
                status="supported",
                method="synthetic_fixture",
                roi=[max(0.0, x - 0.08), max(0.0, y - 0.08), min(1.0, x + 0.08), min(1.0, y + 0.08)],
            )
        if self.uncertain_review:
            if evidence["status"] == "needs_identification":
                evidence.update(
                    status="uncertain",
                    method="seek_new_view_after_inconclusive_identification",
                    previous_image_review=self.uncertain_review,
                )
            self.uncertain_review = None
        self.waiting = True
        self.send("episode_observe", proposal=proposal, evidence=evidence)

    def reconcile(self, now):
        while (event := self.executor.poll()) is not None:
            kind = event["type"]
            if event.get("nonce") != self.nonce:
                continue
            if kind == "episode_state":
                self.episode = event["episode"]
                self.waiting = False
                if self.episode["state"] == "TERMINAL":
                    write_json(self.app.store.root / "episode_summary.json", self.episode)
                    if self.app.trial is not None:
                        self.app.end_trial(self.episode["outcome"], now)
            elif kind == "result":
                result = event["result"]
                self.command_sends += len(result["command_send_statuses"])
                if self.app.scene and self.episode["attempts"]:
                    self.app.scene.settled(self.episode["attempts"][-1]["direction"], result)
                self.cleanup_uncertain = (
                    self.live
                    and result["started_monotonic_s"] is not None
                    and not all(
                        result.get(k) is True
                        for k in ("observed_stationary", "ownership_released", "post_release_stationary")
                    )
                )
            elif kind == "episode_refusal":
                self.last_error = event["reason"]
                self.waiting = False
        if self.executor.process.poll() is not None and self.episode["state"] != "TERMINAL":
            self.cleanup_uncertain = self.live
            self.episode = dict(
                self.episode,
                state="TERMINAL",
                outcome="SENSOR_OR_CONTROL_FAILURE",
                reason="executor_disconnected_cleanup_unverified",
                cleanup_unverified=self.cleanup_uncertain,
            )
            write_json(self.app.store.root / "episode_summary.json", self.episode)
            if self.app.trial:
                self.app.end_trial(self.episode["outcome"], now)

    def process(self, now):
        self.reconcile(now)
        if self.quit_requested:
            self.app.quit, self.app.termination = True, "operator_quit"
        while not self.queue.empty():
            p, queued = self.queue.get_nowait()
            try:
                if p.get("action") == "authorize":
                    self.authorize(p, queued)
                elif p.get("action") in ("identify", "reject", "uncertain", "goal"):
                    self.review_image(p)
                else:
                    raise Hold("unknown_episode_ui_command")
                self.last_error = None
            except (KeyError, TypeError, ValueError, OSError) as exc:
                self.last_error = str(exc)
                self.app.store.event("episode_ui_refusal", now, reason=str(exc))
        if self.waiting or self.abort.is_set() or self.episode["state"] != "OBSERVING":
            return
        pred = self.app._current_prediction(now)
        if pred and pred["request_sequence"] > self.processed_sequence:
            try:
                self.submit_prediction(pred)
            except (KeyError, TypeError, ValueError, OSError) as exc:
                self.waiting = True
                self.send("episode_sensor_failure", reason=str(exc))


class EpisodeApp(InteractiveApp):
    page = episode_page()
    next_messages = {
        **InteractiveApp.next_messages,
        "ready": "Enter the requested object, inspect the displayed area and authorize once in the browser.",
        "started": "One episode is active. Use STOP EPISODE or type stop to end it immediately.",
        "answer": "The browser shows target evidence and the executor decision for this image.",
    }
    trial_end_notice = "See episode_summary.json and executor events for motion and settling."
    prediction_execution_note = (
        "Native proposal. Executed primitives are recorded by episode and prediction ID."
    )
    terminal_help = (
        "Browser authorizes the episode. Terminal accepts stop, status, mark, note, review and quit."
    )

    def __init__(self, *, capability, executor, live, nonce, settings, scene, **kwargs):
        super().__init__(**kwargs)
        self.session.bounded_context = True
        self.scene = scene
        fixture = scene.fixture if scene else None
        self.navigation = EpisodeUI(self, capability, executor, live, nonce, settings, fixture)
        self.primed_epoch = None
        if not live:
            self.session.worker = EpisodeMockWorker(fixture)
        self.banner = (
            "SUPERVISED OBJECT SEARCH AND APPROACH" if live else "SYNTHETIC EPISODE · MOCK DECISIONS"
        )
        self.quiet_failure = None

    def allow_model_request(self, now):
        n, s = self.navigation, self.session
        if n.waiting or n.abort.is_set() or n.episode["state"] != "OBSERVING" or not s.latest:
            return False
        if s.displayed and s.displayed[0].context["identity"]["sequence"] > n.processed_sequence:
            return False
        meta = s.latest
        if self.scene and meta["provenance"].get("synthetic_fixture_view") != self.scene.view:
            return False
        if meta["clock_epoch"] != n.grant["clock_epoch"]:
            return False
        guard = (n.executor.latest_guards or {}).get("snapshot", {})
        bound = guard.get("capture_uncertainty_bound_s")
        stamp = meta["exposure_monotonic_s"]
        if not n.live and s.replay_anchor:
            source, host, speed = s.replay_anchor
            stamp = host + (meta["observation_s"] - source) / speed
        return stamp is not None and bound is not None and stamp - bound > n.episode["after_monotonic_s"]

    def state(self, now):
        status = super().state(now)
        n = self.navigation
        return dict(
            status,
            banner=self.banner,
            navigation=dict(
                deepcopy(n.episode),
                settings={
                    k: n.settings[k] for k in ("max_moves", "seconds", "alignment_deg", "area_radius_m")
                },
                ui_error=n.last_error,
            ),
            robot_commands_sent=n.command_sends if n.live else 0,
        )

    def end_trial(self, reason, now, *, review=True, keep_lidar=False):
        n = self.navigation
        if self.trial and n.episode["state"] != "TERMINAL":
            if self.quiet_failure is None:
                self.quiet_failure = reason
                n.waiting = True
                n.send("episode_sensor_failure", reason=reason)
            return
        return super().end_trial(reason, now, review=review, keep_lidar=keep_lidar or n.cleanup_uncertain)

    def step(self, now):
        n = self.navigation
        if n.live and not n.grant and n.episode["state"] == "IDLE":
            try:
                delivery = self.camera.get_nowait()
            except Empty:
                pass
            else:
                epoch = delivery.observation.clock_epoch
                if epoch != self.primed_epoch:
                    self.session.receive(delivery, self.monotonic())
                    self.primed_epoch = epoch
        n.process(self.monotonic())
        super().step(self.monotonic())
        n.process(self.monotonic())
        if self.trial and self.lidar is not None and (self.lidar.error or not self.lidar.alive):
            self.end_trial("lidar_recorder_source_stopped", self.monotonic())
        if self.trial and self.lidar_recorder is not None:
            recording = self.lidar_recorder.status()
            if recording["errors"] or recording["chunks_dropped"]:
                self.end_trial("incomplete_episode_lidar_recording", self.monotonic())
        if (
            self.trial
            and n.grant
            and self.session.latest
            and (self.session.history_error or self.session.latest["clock_epoch"] != n.grant["clock_epoch"])
        ):
            self.end_trial(self.session.history_error or "camera_epoch_changed", self.monotonic())

    def handle(self, line, typed_s, viewer, now):
        command = line.strip().lower()
        if command in ("stop", "cancel", "quit", "!quit"):
            self.navigation.enqueue({"action": command.lstrip("!")}, now)
        elif self.dialog or command.split(" ", 1)[0] in ("mark", "note", "review", "save"):
            return super().handle(line, typed_s, viewer, now)
        else:
            self.say(json.dumps(self.navigation.episode) if command == "status" else self.terminal_help)

    def trial_end_fields(self):
        return {
            "robot_commands_sent": self.execution_summary()["robot_commands_sent"],
            "episode_id": self.navigation.nonce,
        }

    def execution_summary(self):
        n = self.navigation
        return {
            "episode": n.episode,
            "robot_commands_sent": None
            if n.live and n.cleanup_uncertain
            else n.command_sends
            if n.live
            else 0,
            "physical_execution_authorized": bool(n.live and n.grant),
        }

    def close(self, now):
        n = self.navigation
        n.executor.cancel()
        closed = n.executor.close()
        n.reconcile(self.monotonic())
        if not closed or n.episode["state"] != "TERMINAL":
            n.cleanup_uncertain = n.live
            n.episode = dict(
                n.episode,
                state="TERMINAL",
                outcome="SENSOR_OR_CONTROL_FAILURE" if not closed else "OPERATOR_ABORT",
                reason="executor_cleanup_unverified" if not closed else "session_closed",
                cleanup_unverified=n.cleanup_uncertain,
            )
        write_json(self.store.root / "episode_summary.json", n.episode)
        super().close(self.monotonic())
