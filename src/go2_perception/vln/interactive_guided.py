"""One-answer guided test, driven from the browser. ADVISORY ONLY — NO ROBOT COMMANDS.

Each test asks the model exactly once and then waits for the operator:

    READY    type a target, confirm the safety statement once per run, press "Ask the model"
    ASKING   the trial collects ~1 s of frames, submits ONE request, and stops archiving frames
    ANSWER   one answer is shown with its own image; the operator picks a verdict and may comment
    SAVED    the verdict is stored (separate human-label file); "Next target" returns to READY

No streaming answers, no automatic end while the operator is deciding. Browser actions are
queued and applied by the main loop, so the HTTP threads never change application state.
"""

import json
from queue import Full, Queue
import threading

from .interactive_trials import append_jsonl, expand_target, utc_now

READY, ASKING, ANSWER, SAVED = "READY", "ASKING", "ANSWER", "SAVED"
SAFETY_STATEMENT = ("The Go2 is stationary, the handheld remote owns control (no app/API control), "
                    "onboard avoidance is on, and body clearance and cable slack are checked.")
POINT_CHOICES = {"correct": "Yes — the + is on the target", "wrong": "No — the + is not on the target",
                 "unsure": "Not sure", "target_not_in_view": "The target is not in view"}
NOT_VISIBLE_CHOICES = {"agree_not_in_view": "Right — the target is not in view",
                       "disagree_target_in_view": "Wrong — the target IS in view",
                       "unsure": "Not sure"}
ASK_TIMEOUT_S = 60.0


def choices_for(prediction):
    if prediction is None:
        return {}
    if prediction.get("target_fraction"):
        return POINT_CHOICES
    if prediction.get("opos_not_visible"):
        return NOT_VISIBLE_CHOICES
    return {"no_point": "The model gave no usable point", "unsure": "Not sure"}


class GuidedTest:
    def __init__(self, app):
        self.app = app
        self.actions = Queue(maxsize=16)
        self.step = READY
        self.target = None
        self.trial_id = None
        self.prediction_id = None
        self.asked_s = None
        self.message = "Type a target and press “Ask the model”."
        self.saved = None
        self.count = 0
        self.lock = threading.Lock()
        self.pending_save = None  # prediction ID of a queued, not yet processed Save

    # ---------- browser thread ----------
    def enqueue(self, payload, now):
        if payload.get("action") not in ("ask", "answer", "cancel", "next", "quit"):
            raise ValueError("Unknown action")
        with self.lock:
            if payload["action"] == "answer":
                if self.pending_save is not None:
                    raise ValueError("a Save is already being processed")
                if not payload.get("prediction_id") or not payload.get("trial_id"):
                    raise ValueError("Save must name the trial and prediction it judges")
            try:
                self.actions.put_nowait((dict(payload), now))
            except Full:
                raise ValueError("Busy; try again") from None
            if payload["action"] == "answer":
                self.pending_save = payload["prediction_id"]

    # ---------- main loop ----------
    def process(self, now):
        while not self.actions.empty():
            payload, _ = self.actions.get_nowait()
            try:
                getattr(self, "_" + payload["action"])(payload, now)
            except ValueError as error:
                self.message = f"Not done: {error}"
            finally:
                if payload["action"] == "answer":
                    with self.lock:
                        self.pending_save = None
        app = self.app
        if (self.step == ANSWER and (app.trial is None or app.trial.id != self.trial_id)
                and "trial was ended" not in self.message):
            self.message = (f"The trial was ended ({self._end_reason()}). Your verdict on THIS image can still "
                            "be saved, or Discard it.")
        if self.step == ASKING:
            pred = next((p for p in reversed(app.predictions.values()) if p["trial_id"] == self.trial_id), None)
            ended = app.trial is None or app.trial.id != self.trial_id
            if pred is not None:
                self.prediction_id, self.step = pred["prediction_id"], ANSWER
                self.message = "Answer ready. Choose what you see, add a comment if you like, then Save."
                app.say(f"[guided] answer ready: {pred['prediction_id']}")
            elif ended:
                # Ended or replaced elsewhere (terminal command, resource limit, source stop).
                self.step = READY
                self.message = (f"That test ended before an answer arrived ({self._end_reason()}). "
                                "Nothing was saved. Ask again when ready.")
            elif app.session.history_error:
                self._fail(f"the model could not run ({app.session.history_error})", now)
            elif now - self.asked_s > ASK_TIMEOUT_S:
                self._fail("no answer within 60 s", now)

    def _end_reason(self):
        try:
            path = self.app.store.trial_dir(self.trial_id) / "trial_end.json"
            return json.loads(path.read_text())["end_reason"]
        except (OSError, ValueError, KeyError, FileNotFoundError):
            return "ended elsewhere"

    def _fail(self, why, now):
        if self.app.trial is not None and self.app.trial.id == self.trial_id:
            self.app.end_trial("guided_failed: " + why, now, review=False)
        self.step, self.message = READY, f"Not done: {why}. Nothing was saved. Try again."

    def _ask(self, payload, now):
        app = self.app
        if self.step not in (READY, SAVED):
            raise ValueError("finish or cancel the current test first")
        target = " ".join(str(payload.get("target", "")).split())
        spec = expand_target(target)
        if app.safety_confirmation is None:
            if payload.get("safety_confirmed") is not True:
                raise ValueError("tick the safety statement first")
            app.safety_confirmation = {"answer": "yes", "statement": SAFETY_STATEMENT, "wall_utc": utc_now(),
                                       "via": "browser checkbox",
                                       "note": "operator confirmation; the app cannot verify it and sends nothing"}
            app.store.event("safety_question_confirmed", now, **app.safety_confirmation)
        trial = app.start_trial(spec, now, single_answer=True)
        if trial is None:
            raise ValueError("the trial could not start (see the terminal log)")
        self.count += 1
        self.target, self.trial_id, self.prediction_id = spec["operator_text"], trial.id, None
        self.asked_s, self.saved = now, None
        self.step, self.message = ASKING, "Asking the model… keep the robot still."

    def _answer(self, payload, now):
        app = self.app
        if self.step != ANSWER:
            raise ValueError("there is no answer to judge (it may already be saved or cancelled)")
        if payload.get("trial_id") != self.trial_id or payload.get("prediction_id") != self.prediction_id:
            raise ValueError("this Save refers to an older answer; the page now shows a different one")
        pred = app.predictions.get(self.prediction_id)
        image = (pred or {}).get("source_image") or {}
        if not image.get("file_sha256") or payload.get("image_sha256") != image["file_sha256"]:
            raise ValueError("the image you judged does not match the archived model input; reload and retry")
        choices = choices_for(pred)
        choice = payload.get("choice")
        if choice not in choices:
            raise ValueError("choose one of the buttons")
        comment = str(payload.get("comment", ""))
        if len(comment) > 2000:
            raise ValueError("comment too long (2000 characters max)")
        record = {
            "schema": "go2-vln-guided-answer-review-v1", "trial_id": self.trial_id,
            "prediction_id": self.prediction_id, "target_operator_text": self.target,
            "question": "Is the + on the target?" if pred and pred.get("target_fraction")
            else "The model says the target is NOT VISIBLE. Is that right?",
            "choice": choice, "choice_text": choices[choice], "verbatim_comment": comment,
            "reviewer": app.config.get("reviewer"), "source_image": image,
            "displayed_image_sha256": payload["image_sha256"],
            "observation_age_at_review_s": self._age_now(pred, now),
            "trial_already_ended_before_save": app.trial is None or app.trial.id != self.trial_id,
            "prediction_label_at_receipt": pred and pred.get("label_at_receipt"),
            "reviewed_wall_utc": utc_now(), "reviewed_host_monotonic_s": now,
            "confirmed_by_user": True, "source": "human_operator", "model_feedback": False,
        }
        append_jsonl(app.store.trial_dir(self.trial_id) / "human_labels.jsonl", record)
        app.store.event("guided_answer_saved", now, trial_id=self.trial_id, prediction_id=self.prediction_id,
                        choice=choice)
        if app.trial is not None and app.trial.id == self.trial_id:
            app.end_trial("guided_answer_saved", now, review=False)
        self.saved = {"choice_text": choices[choice], "prediction_id": self.prediction_id}
        self.step = SAVED
        self.message = f"Saved: “{choices[choice]}”. Type the next target, or press “Close tool”."

    def _cancel(self, payload, now):
        if self.step not in (ASKING, ANSWER):
            raise ValueError("nothing to cancel")
        if self.app.trial is not None and self.app.trial.id == self.trial_id:
            self.app.end_trial("guided_cancelled", now, review=False)
        self.step, self.message = READY, "Cancelled. Nothing was saved for that test."

    def _quit(self, payload, now):
        # Closes this read-only tool only (same as typing `quit`); nothing is sent to the robot.
        self.app.quit, self.app.termination = True, "operator_quit_from_browser"
        self.message = "Closing the tool…"

    def _next(self, payload, now):
        self.step, self.message = READY, "Type a target and press “Ask the model”."

    @staticmethod
    def _age_now(pred, now):
        received = (pred or {}).get("source_frame", {}).get("received_monotonic_s")
        return None if received is None else now - received

    def public(self, now):
        pred = self.app.predictions.get(self.prediction_id) if self.prediction_id else None
        return {
            "step": self.step, "message": self.message, "target": self.target, "trial_id": self.trial_id,
            "prediction_id": self.prediction_id, "test_number": self.count,
            "asking_for_s": None if self.step != ASKING else now - self.asked_s,
            "safety_needed": self.app.safety_confirmation is None, "safety_statement": SAFETY_STATEMENT,
            "choices": choices_for(pred) if self.step == ANSWER else {},
            "answer": None if pred is None else {
                "prediction_id": pred["prediction_id"], "target_fraction": pred.get("target_fraction"),
                "opos_not_visible": pred.get("opos_not_visible"), "label_at_receipt": pred["label_at_receipt"],
                "inference_duration_s": pred.get("inference_duration_s"), "mock": pred.get("mock"),
                "trial_id": pred["trial_id"],
                "image_sha256": (pred.get("source_image") or {}).get("file_sha256"),
                "age_at_receipt_s": (pred.get("display_timing_at_receipt") or {}).get("host_since_delivery_s"),
                "exposure_age_at_receipt_s": (pred.get("display_timing_at_receipt") or {}).get(
                    "observation_age_at_display_s"),
                "age_now_s": self._age_now(pred, now),
                "image_url": f"/api/pred_image?id={pred['prediction_id']}" if pred.get("source_image") else None,
                "source_frame_id": pred["source_frame"]["frame_id"]},
            "saved": self.saved,
            "pending_save": self.pending_save,
        }
