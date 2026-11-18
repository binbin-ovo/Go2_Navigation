"""One-answer guided test, driven from the browser. ADVISORY ONLY — NO ROBOT COMMANDS.

Each test asks the model exactly once and then waits for the operator:

    READY    type a target, confirm the safety statement once per run, press "Ask the model"
    ASKING   the trial collects ~1 s of frames, submits ONE request, and stops archiving frames
    ANSWER   one answer is shown with its own image; the operator picks a verdict and may comment
    SAVED    the verdict is stored (separate human-label file); "Next target" returns to READY

No streaming answers, no automatic end while the operator is deciding. Browser actions are
queued and applied by the main loop, so the HTTP threads never change application state.
"""

from queue import Full, Queue

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

    # ---------- browser thread ----------
    def enqueue(self, payload, now):
        if payload.get("action") not in ("ask", "answer", "cancel", "next", "quit"):
            raise ValueError("Unknown action")
        try:
            self.actions.put_nowait((dict(payload), now))
        except Full:
            raise ValueError("Busy; try again") from None

    # ---------- main loop ----------
    def process(self, now):
        while not self.actions.empty():
            payload, _ = self.actions.get_nowait()
            try:
                getattr(self, "_" + payload["action"])(payload, now)
            except ValueError as error:
                self.message = f"Not done: {error}"
        app = self.app
        if self.step == ASKING and app.trial is not None and app.trial.id == self.trial_id:
            pred = next((p for p in reversed(app.predictions.values()) if p["trial_id"] == self.trial_id), None)
            if pred is not None:
                self.prediction_id, self.step = pred["prediction_id"], ANSWER
                self.message = "Answer ready. Choose what you see, add a comment if you like, then Save."
                app.say(f"[guided] answer ready: {pred['prediction_id']}")
            elif app.session.history_error:
                self._fail(f"the model could not run ({app.session.history_error})", now)
            elif now - self.asked_s > ASK_TIMEOUT_S:
                self._fail("no answer within 60 s", now)

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
            raise ValueError("there is no answer to judge")
        pred = app.predictions.get(self.prediction_id)
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
            "reviewer": app.config.get("reviewer"), "source_image": pred and pred.get("source_image"),
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
                "image_url": f"/api/pred_image?id={pred['prediction_id']}" if pred.get("source_image") else None,
                "source_frame_id": pred["source_frame"]["frame_id"]},
            "saved": self.saved,
        }
