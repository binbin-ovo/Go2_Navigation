"""Opt-in supervised extension of the existing four-panel viewer; defaults unchanged."""
from copy import deepcopy
import hashlib
import json
from queue import Full, Queue
import secrets
import time

import numpy as np

from .interactive_app import InteractiveApp
from .interactive_page import PAGE
from .interactive_trials import expand_target
from .shadow_sources import SyntheticSource
from .shadow_worker import MockWorker
from .supervised import Cycle, Hold, MODEL_REVISION, contract_version
from .supervised_target import correspondence


class NavigationMockWorker(MockWorker):
    def poll(self, now):
        request = self.pending[0] if self.pending else None
        response = super().poll(now)
        if response is not None:
            response["result"]["pointing"]["opos"] = 13*48+24
            response["result"]["native_output"] = response["result"]["native_output"].replace("opos_0", "opos_648")
            (request.folder / "mock_result.json").write_text(json.dumps(response["result"], indent=2)+"\n")
        return response


class StationaryMockSource(SyntheticSource):
    """Textured stationary invented image; no claim of semantic inference or motion rendering."""
    def __iter__(self):
        rng = np.random.default_rng(672)
        image = rng.integers(0, 256, (256, 448, 3), dtype=np.uint8)
        for delivery in super().__iter__():
            delivery.observation.visual.image[:] = image
            yield delivery


CONTROLS = """<style>#test{display:none!important}#navstatus{white-space:pre-wrap}#pimg{touch-action:none}</style>
<section id="navigation"><h2>One supervised bounded step</h2><strong id="navprofile"></strong>
<p>Independent short observation for preview and refresh. No continuous model memory.</p>
<div class="row"><input id="navtarget" type="text" placeholder="Target, e.g. the chair" maxlength="160">
<button id="navask">Obtain preview</button><button id="navquit">Close session</button></div>
<p>Drag a box around the physical target in panel B, then confirm its instance.</p>
<div class="row"><input id="navinstance" type="text" placeholder="Instance, e.g. chair beside doorway" maxlength="160">
<input id="navoperator" type="text" placeholder="Operator name" maxlength="100">
<button id="navconfirm">Confirm target and bound action</button></div>
<p><label><input id="navsafe" type="checkbox">I checked the bounded action, target instance, clear static zone,
cable slack, independent intervention and exclusive command source for this ONE cycle.</label></p>
<div class="row"><button id="navinspect" hidden>I inspected the full 0.65 m radius, blind rear and cable for this turn</button></div>
<div class="row"><button id="navauth">Authorize ONE refresh/action</button>
<button id="navcancel">Cancel pending proposal</button><button id="navstop">Request STOP from executor</button></div>
<p id="navstatus" role="status"></p><pre id="navmeasure"></pre>
<details><summary>Raw decision-frame waypoints and selected body primitive</summary><pre id="navpath"></pre></details></section>"""
JAVASCRIPT = r"""
let navROI=null,navDrag=null,navImageReady=null,navPred=null;
async function navSend(body){const r=await fetch('/api/navigation',{method:'POST',headers:{'Content-Type':'application/json','X-Review-Token':TOKEN},body:JSON.stringify(body)});if(!r.ok)alert((await r.json()).error||'Request refused')}
$('navask').onclick=()=>navSend({action:'ask',target:$('navtarget').value});
$('navquit').onclick=()=>navSend({action:'quit'});
$('navconfirm').onclick=()=>navSend({action:'confirm',prediction_id:navPred,source_image_sha256:navImageReady,roi:navROI,instance:$('navinstance').value,operator:$('navoperator').value});
$('navinspect').onclick=()=>navSend({action:'inspect_turn',prediction_id:navPred,source_image_sha256:navImageReady,operator:$('navoperator').value});
$('navauth').onclick=()=>{navSend({action:'authorize',prediction_id:navPred,source_image_sha256:navImageReady,safety_confirmed:$('navsafe').checked});$('navsafe').checked=false};
$('navcancel').onclick=()=>navSend({action:'cancel'});
$('navstop').onclick=()=>navSend({action:'stop'});
function navXY(e){const r=$('pimg').getBoundingClientRect();return [Math.max(0,Math.min(1,(e.clientX-r.left)/r.width)),Math.max(0,Math.min(1,(e.clientY-r.top)/r.height))]}
$('pimg').addEventListener('pointerdown',e=>{if(!navImageReady)return;navDrag=navXY(e);$('pimg').setPointerCapture(e.pointerId)});
$('pimg').addEventListener('pointerup',e=>{if(!navDrag)return;const p=navXY(e);navROI=[Math.min(navDrag[0],p[0]),Math.min(navDrag[1],p[1]),Math.max(navDrag[0],p[0]),Math.max(navDrag[1],p[1])];navDrag=null;const c=$('pimg'),g=c.getContext('2d');g.strokeStyle='#ff00ff';g.lineWidth=2;g.strokeRect(navROI[0]*c.width,navROI[1]*c.height,(navROI[2]-navROI[0])*c.width,(navROI[3]-navROI[1])*c.height)});
function panelNavigation(){const n=S.navigation||{},busy=['EXECUTING','STOPPING'].includes(n.state);$('banner').textContent=S.banner;
 if(navPred!==(S.prediction||{}).prediction_id){navImageReady=null;navROI=null;navPred=(S.prediction||{}).prediction_id}
 const a=n.action||{},seg=a.segment||{},r=n.result||{},m=r.motion_measurement||{},f=n.feedback||{};
 const inspected=n.turn_scope==='operator_inspected_static_disc'&&a.type==='turn';
 $('navinspect').hidden=!inspected||n.state!=='REVIEW';$('navinspect').disabled=n.turn_inspected;
 $('navprofile').textContent=(n.motion_profile||'')+' | Forward '+(n.rates||{}).forward_mps+' m/s | Turn '+(n.rates||{}).turn_radps+' rad/s';
 $('navstatus').textContent=n.state+' — '+n.reason;
 const fmt=x=>Number.isFinite(x)?x.toFixed(3):'unavailable';
 const lines=[];
 if(a.type)lines.push(a.direction+' '+fmt(a.magnitude)+' '+a.unit+' at '+fmt(a.speed)+' '+a.speed_unit+' | deadline '+a.duration_s+' s');
 if(seg.policy)lines.push('Feedback trigger '+fmt(seg.stop_trigger)+' '+a.unit+' | required clearance '+fmt(seg.required_clearance)+' '+a.unit+' | scope '+n.turn_scope);
 if(f.phase)lines.push('Feedback '+f.phase+' '+fmt(f.value)+' '+f.unit);
 if(m.outcome)lines.push(m.outcome+' | trigger sample '+fmt(m.at_first_zero)+' | settled '+fmt(m.settled)+' '+m.unit+' | '+m.termination);
 if(r.status)lines.push('Execution '+r.status+' — '+r.reason+' | stationary '+r.observed_stationary+' | release '+r.release_rpc_code+' | still after release '+r.post_release_stationary);
 if(n.replan)lines.push('Post-turn advisory '+(n.replan.refusal||n.replan.candidate.direction)+' | separate authorization required for movement');
 $('navmeasure').textContent=lines.join('\n');
 const rows=n.native_waypoints||[], marked=(seg.mapped_prefix||[]).map(x=>x.row);
 $('navpath').textContent='Native rows: x forward (m), y left (m), yaw CCW (rad)\n'+rows.map((row,i)=>i+'  '+row.map(fmt).join('  ')+(marked.includes(i)?(i===a.selected_row?'  endpoint bracket × '+fmt(seg.end_fraction):'  checked prefix'):'')).join('\n')+
 (a.type?'\nSelected body primitive: '+a.direction+' '+fmt(a.magnitude)+' '+a.unit:'');
 $('navask').disabled=!['IDLE','HOLD'].includes(n.state)||n.locked;
 $('navconfirm').disabled=!navImageReady||!navROI||!['IDLE','HOLD','REVIEW'].includes(n.state)||n.locked;
 $('navauth').disabled=n.state!=='REVIEW'||!navImageReady||(inspected&&!n.turn_inspected);$('navcancel').disabled=busy;$('navstop').disabled=!busy;
}
"""


def navigation_page():
    page = PAGE.replace('<section id="test">', CONTROLS + '<section id="test">', 1)
    page = page.replace('ADVISORY ONLY — NO ROBOT COMMANDS', 'SUPERVISED ONE-STEP SESSION — CHECK MODE')
    page = page.replace('function drawPred(p,img){',
                        'function drawPred(p,img){if(!S.prediction||p.prediction_id!==S.prediction.prediction_id)return;'
                        'navPred=p.prediction_id;navROI=null;navImageReady=img&&p.source_image?p.source_image.file_sha256:null;')
    page = page.replace('strip();panelTest();', 'strip();panelNavigation();')
    return page.replace('poll();pollLidar();', JAVASCRIPT + '\npoll();pollLidar();')


class NavigationUI:
    def __init__(self, app, cycle, executor):
        self.app, self.cycle, self.executor = app, cycle, executor
        self.queue = Queue(maxsize=16)
        self.expected_trial = None
        self.seen = set()
        self.instruction_id = None
        self.preview_prediction = None
        self.preview_proposal = None
        self.review_path = None
        self.intent = None
        self.refresh_started = None
        self.pending_refresh = None

    def enqueue(self, payload, now):
        if payload.get("action") not in ("ask", "confirm", "inspect_turn", "authorize", "cancel", "stop", "quit"):
            raise ValueError("unknown navigation action")
        # HTTP cancel/stop is independent of model preparation and the UI event loop.
        if payload["action"] in ("cancel", "stop", "quit"):
            self.executor.cancel()
        try:
            self.queue.put_nowait((deepcopy(payload), now))
        except Full:
            raise ValueError("navigation queue full; cancellation was still sent immediately") from None

    def _start(self, spec, now, intent):
        trial = self.app.start_trial(spec, now, single_answer=True)
        if trial is None:
            raise Hold("model_or_source_not_ready")
        self.expected_trial, self.intent = trial.id, intent
        # The camera inbox can contain pre-authorization deliveries. Drain ALL of it;
        # do not let a queued old frame become part of the fresh model request.
        while not self.app.tap.queue.empty():
            self.app.tap.queue.get_nowait()

    def _identity(self, pred):
        frame, image = pred["source_frame"], pred["source_image"]
        observed = frame.get("exposure_monotonic_s")
        if not self.cycle.live:
            anchor = self.app.session.replay_anchor
            if anchor is None:
                raise Hold("synthetic_clock_mapping_unavailable")
            observed = anchor[1] + (frame["observation_s"] - anchor[0])/anchor[2]
        if observed is None:
            raise Hold("capture_monotonic_mapping_unavailable")
        guard = self.executor.guard_at(observed, self.cycle.cap["limits"]["state_age_s"])
        if self.cycle.live and (guard.get("capture_clock_epoch") != frame["clock_epoch"] or
                                not guard.get("capture_bound_basis")):
            raise Hold("capture_age_bound_not_validated_for_this_camera_epoch")
        return {"session_id": self.app.session.session_id, "trial_id": pred["trial_id"],
                "instruction_id": self.instruction_id, "instruction": pred["instruction"],
                "prediction_id": pred["prediction_id"], "observation_id": frame["frame_id"],
                "clock_epoch": frame["clock_epoch"], "source_image_sha256": image["file_sha256"],
                "observed_monotonic_s": observed, "received_monotonic_s": frame["received_monotonic_s"],
                "uncertainty_bound_s": guard.get("capture_uncertainty_bound_s"),
                "pose": guard["pose"], "geometry": guard["geometry"]}

    def proposal(self, pred):
        if not pred.get("source_image"):
            raise Hold("archived_source_image_unavailable")
        path = self.app.store.root / pred["source_image"]["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != pred["source_image"]["file_sha256"]:
            raise Hold("source_image_changed")
        if pred["waypoint_convention"] != "absolute_ego_SE2_x_forward_y_left_yaw_CCW_metres_radians":
            raise Hold("unsupported_waypoint_convention")
        if pred["rejection_reason_at_receipt"]:
            raise Hold("model_result_rejected: " + pred["rejection_reason_at_receipt"])
        if self.cycle.live:
            identity_path = self.app.store.root / "session/worker/model.json"
            identity = json.loads(identity_path.read_text())
            # Keep source identity in archives; require the pinned checkpoint in its identity.
            if identity.get("checkpoint") != MODEL_REVISION or identity.get("upstream_code") != "3015508b70fb6e30bbd66701e94c9b916dcb8df8" or not identity.get("processor"):
                raise Hold("pinned_model_identity_not_verified")
        revision = MODEL_REVISION if self.cycle.live else "synthetic"
        proposal = {"identity": self._identity(pred), "native_output": pred["native_output"],
                    "model_identity": {"checkpoint": "LightOriginsHQ/LightNav-0" if self.cycle.live else "synthetic-mock",
                                       "revision": revision, "processor_revision": revision},
                    "waypoints": pred["waypoints"], "native_stop": pred["native_stop"],
                    "point_fraction": None if pred["target_fraction"] is None else list(pred["target_fraction"]),
                    "synthetic": pred["mock"]}
        return proposal, path

    def process(self, now):
        while not self.queue.empty():
            payload, queued = self.queue.get_nowait()
            try:
                self.command(payload, now, queued)
            except (ValueError, KeyError, TypeError, OSError) as error:
                if self.cycle.state in ("EXECUTING", "STOPPING"):
                    self.executor.cancel()
                    self.cycle.cancel()
                    self.cycle.reason += " Rejected UI request: " + str(error)
                else:
                    self.cycle.hold(error)
                self.app.say("HOLD: " + str(error))
        if self.pending_refresh is not None:
            if self.cycle.state != "REFRESHING" or now >= self.cycle.authorization["expires_monotonic_s"]:
                self.pending_refresh = None
            else:
                item = self.app.tap.frame(self.app.tap.latest_id)
                if item is not None:
                    meta = item[1]
                    stamp = meta.get("exposure_monotonic_s")
                    guard = (self.executor.latest_guards or {}).get("snapshot", {})
                    bound = guard.get("capture_uncertainty_bound_s")
                    if (stamp is not None and bound is not None and guard.get("capture_bound_basis") and
                            guard.get("capture_clock_epoch") == meta["clock_epoch"] and stamp-bound > self.refresh_started):
                        spec, self.pending_refresh = self.pending_refresh, None
                        try:
                            self._start(spec, now, "refresh")
                        except Hold as error:
                            self.cycle.hold(error)
        if self.intent:
            pred = next((p for p in reversed(self.app.predictions.values()) if p["trial_id"] == self.expected_trial), None)
            if pred and pred["prediction_id"] not in self.seen:
                self.seen.add(pred["prediction_id"])
                try:
                    if self.intent in ("refresh", "post_turn"):
                        proposal, path = self.proposal(pred)
                        snapshot = json.loads((self.app.store.root / pred["request_directory"] / "snapshot.json").read_text())
                        # Every causal source in the independent refresh must follow approval.
                        for entry in snapshot["history"]:
                            source = entry["source"]
                            stamp = source.get("exposure_monotonic_s")
                            if not self.cycle.live:
                                anchor = self.app.session.replay_anchor
                                stamp = anchor[1] + (source["observation_s"]-anchor[0])/anchor[2]
                            if self.cycle.live and contract_version(self.cycle.cap) == 2:
                                guard = self.executor.guard_at(stamp, self.cycle.cap["limits"]["state_age_s"]) if stamp is not None else {}
                                bound = guard.get("capture_uncertainty_bound_s")
                                if (bound is None or not guard.get("capture_bound_basis") or
                                        guard.get("capture_clock_epoch") != source["clock_epoch"]):
                                    raise Hold("unverified_history_capture_bound")
                                stamp -= bound
                            if stamp is None or stamp <= self.refresh_started:
                                raise Hold("refresh_history_contains_pre_authorization_frame")
                        if self.intent == "post_turn":
                            self.cycle.record_replan(proposal)
                            out = self.app.store.root / "navigation/post_turn_advisory.json"
                            out.write_text(json.dumps(self.cycle.replan, indent=2)+"\n")
                            (out.parent/"cycle_result.json").write_text(json.dumps(self.cycle.public(now), indent=2)+"\n")
                            self.app.end_trial("post_turn_read_only_replan", now, review=False)
                        else:
                            comparison = correspondence(self.review_path, path, self.cycle.target, proposal["point_fraction"])
                            current = self.executor.latest_guards
                            if current is None:
                                raise Hold("executor_guards_unavailable")
                            ticket = self.cycle.refreshed(proposal, comparison, now=self.app.monotonic(), wall_s=time.time(),
                                                          current=current["snapshot"])
                            self.executor.submit(ticket)
                            self.app.end_trial("one_refresh_dispatched", now, review=False)
                    else:
                        self.preview_prediction = pred
                        self.preview_proposal, self.review_path = self.proposal(pred)
                        self.cycle.reason = "Drag target ROI on panel B; confirm physical instance to create bounded preview."
                except (ValueError, KeyError, TypeError, OSError) as error:
                    self.cycle.hold(error)
                self.intent = None
        if self.cycle.state == "REPLANNING" and self.intent and (not self.app.trial or now-self.refresh_started > 30.):
            self.intent = None
            self.cycle.hold("post_turn_advisory_unavailable_or_timed_out")
            if self.app.trial:
                self.app.end_trial("post_turn_advisory_timeout", now, review=False)
        if self.cycle.state == "REFRESHING" and now >= self.cycle.authorization["expires_monotonic_s"]:
            self.cycle.hold("refresh_timeout_authorization_spent")
            self.intent = None
            if self.app.trial:
                self.app.end_trial("refresh_timeout", now, review=False)
        while (message := self.executor.poll()) is not None:
            self.app.store.event("navigation_executor", now, message=message)
            if message["type"] == "result":
                try:
                    self.cycle.completed(message["result"])
                except Hold as error:
                    self.cycle.locked = True
                    self.cycle.hold(error)
            elif message["type"] == "feedback":
                if message.get("nonce") == self.cycle.nonce and message.get("request_id") == self.cycle.request_id:
                    self.cycle.feedback = {k: message[k] for k in ("phase", "value", "unit", "monotonic_s")}
            elif message["type"] in ("fatal", "intervention_required"):
                self.cycle.locked = True
                self.cycle.hold(message.get("error") or message.get("reason"))
        if self.cycle.state == "AWAITING_OBSERVATION":
            try:
                self._post_observation(now)
            except OSError as error:
                self.cycle.locked = True
                self.cycle.hold("post_motion_archive_failed: "+str(error))
            except Hold as error:
                # A newer guard can arrive on the next iteration; never run inference/motion.
                if now-self.cycle.result["ended_monotonic_s"] > 3:
                    self.cycle.locked = True
                    self.cycle.hold("post_motion_observation_unverified: " + str(error))
        if self.cycle.state in ("EXECUTING", "STOPPING") and self.executor.process.poll() is not None:
            self.cycle.locked = True
            self.cycle.hold("executor_lost_stop_and_release_unknown")

    def command(self, p, now, queued):
        action = p["action"]
        if action in ("cancel", "stop", "quit"):
            self.cycle.cancel()
            self.intent = None
            if self.app.trial:
                self.app.end_trial("navigation_cancelled", now, review=False)
            if action == "quit":
                self.app.quit, self.app.termination = True, "operator_closed_supervised_session"
            return
        if now-queued > .5:
            raise Hold("queued_operator_action_expired")
        if action == "ask":
            if self.cycle.state not in ("IDLE", "HOLD") or self.cycle.locked or self.cycle.dispatched:
                raise Hold("session_busy_or_one_cycle_already_executed")
            self.cycle.last_proposal = self.cycle.action = self.cycle.reviewed = self.cycle.target = self.cycle.inspection = None
            self.instruction_id = secrets.token_hex(16)
            self.preview_prediction = self.preview_proposal = None
            self.cycle.state = "IDLE"
            self._start(expand_target(p["target"]), now, "preview")
        elif action == "confirm":
            pred = self.preview_prediction
            if not pred or pred["prediction_id"] != p.get("prediction_id") or pred["source_image"]["file_sha256"] != p.get("source_image_sha256"):
                raise Hold("target_confirmation_names_another_image")
            proposal = self.preview_proposal
            if proposal is None or hashlib.sha256(self.review_path.read_bytes()).hexdigest() != p["source_image_sha256"]:
                raise Hold("reviewed_image_unavailable_or_changed")
            target = {"instance_id": str(p["instance"]).strip(), "reviewed_image_sha256": p["source_image_sha256"],
                      "roi_xyxy": p["roi"], "confirmed_by": str(p["operator"]).strip(), "confirmed_monotonic_s": now}
            self.cycle.preview(proposal, target, now=now, wall_s=time.time())
        elif action == "inspect_turn":
            identity = (self.cycle.reviewed or {}).get("identity", {})
            if (identity.get("prediction_id"), identity.get("source_image_sha256")) != (p.get("prediction_id"), p.get("source_image_sha256")):
                raise Hold("inspection_for_wrong_preview")
            current = self.executor.latest_guards
            if not current:
                raise Hold("executor_guards_unavailable")
            self.cycle.inspect_turn(operator=p.get("operator"), now=now, current=current["snapshot"])
        elif action == "authorize":
            if p.get("safety_confirmed") is not True or self.cycle.reviewed is None:
                raise Hold("one_cycle_operator_confirmation_required")
            old = self.cycle.reviewed["identity"]
            if (p.get("prediction_id"), p.get("source_image_sha256")) != (old["prediction_id"], old["source_image_sha256"]):
                raise Hold("authorization_for_wrong_preview")
            self.cycle.authorize(now=now, wall_s=time.time())
            self.refresh_started = now
            spec = {"instruction": old["instruction"], "operator_text": old["instruction"],
                    "template_id": None, "kind": "instruction"}
            if self.cycle.live and contract_version(self.cycle.cap) == 2:
                # Start independent history only once even the oldest possible
                # capture of a newly delivered camera frame follows approval.
                self.pending_refresh = spec
            else:
                self._start(spec, now, "refresh")

    def _post_observation(self, now):
        item = self.app.tap.frame(self.app.tap.latest_id)
        if item is None:
            raise Hold("post_motion_camera_unavailable")
        image, meta = item
        observed = meta.get("exposure_monotonic_s")
        if not self.cycle.live:
            stamp = meta["stamp_robot"]
            anchor = self.app.session.replay_anchor
            observed = anchor[1] + (stamp["t"]-anchor[0])/anchor[2]
        if observed is None or observed <= self.cycle.result["ended_monotonic_s"]:
            if now-self.cycle.result["ended_monotonic_s"] > 3:
                raise Hold("post_motion_camera_missing_or_no_capture_mapping")
            return
        import cv2
        path = self.app.store.root / "navigation/post_motion.png"
        path.parent.mkdir(exist_ok=True)
        if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
            raise OSError("post-motion image save failed")
        identity = deepcopy(self.cycle.execution_identity)
        guard = self.executor.guard_at(observed, self.cycle.cap["limits"]["state_age_s"])
        if self.cycle.live and contract_version(self.cycle.cap) == 2:
            bound = guard.get("capture_uncertainty_bound_s")
            if (bound is None or not guard.get("capture_bound_basis") or guard.get("capture_clock_epoch") != meta["clock_epoch"] or
                    observed-bound <= self.cycle.result["ended_monotonic_s"]):
                raise Hold("post_motion_capture_interval_unverified")
            identity["uncertainty_bound_s"] = bound
        identity.update(prediction_id="post-motion-observation-no-model", observation_id=meta["frame_id"],
                        source_image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), clock_epoch=meta["clock_epoch"],
                        observed_monotonic_s=observed, received_monotonic_s=meta["received_monotonic_s"],
                        pose=guard["pose"], geometry=guard["geometry"])
        self.cycle.observe_after(identity)
        (path.parent / "post_motion_source.json").write_text(json.dumps({"source": meta,
            "image_kind": "new camera tap frame, not a new model proposal", "source_image_sha256": identity["source_image_sha256"]}, indent=2)+"\n")
        (path.parent / "cycle_result.json").write_text(json.dumps(self.cycle.public(now), indent=2)+"\n")
        archive = {"schema": f"go2-supervised-navigation-v{contract_version(self.cycle.cap)}", "synthetic": not self.cycle.live,
                   "capability": self.cycle.cap, "proposal": self.cycle.dispatched["proposal"],
                   "authorization": self.cycle.dispatched["authorization"],
                   "execution_result": self.cycle.result, "post_observation": self.cycle.post}
        if contract_version(self.cycle.cap) == 2:
            archive["executed_action"] = self.cycle.dispatched["action"]
        (path.parent / "cycle_record.json").write_text(json.dumps(archive, indent=2)+"\n")
        if self.cycle.state == "REPLANNING":
            self.refresh_started = self.cycle.result["ended_monotonic_s"]
            try:
                self._start({"instruction": identity["instruction"], "operator_text": identity["instruction"],
                             "template_id": None, "kind": "instruction"}, now, "post_turn")
            except Hold as error:
                self.cycle.hold("post_turn_advisory_unavailable: "+str(error))


class SupervisedApp(InteractiveApp):
    page = navigation_page()
    trial_end_notice = "Model observation ended. Executor motion/stop/release status is reported separately."
    terminal_help = "Use the supervised browser controls. Terminal: status | cancel pending | stop active | quit."

    def __init__(self, *, capability, executor, live, nonce, **kwargs):
        super().__init__(**kwargs)
        self.banner = ("LIVE SUPERVISED — ONE OPERATOR-AUTHORIZED BOUNDED ACTION" if live else
                       "SYNTHETIC MOCK — NO ROBOT CONNECTION OR COMMANDS")
        self.cycle = Cycle(capability, live=live, nonce=nonce,
                           event=lambda kind, **fields: self.store.event(kind, self.monotonic(), **fields))
        self.navigation = NavigationUI(self, self.cycle, executor)
        if not live:
            self.session.worker = NavigationMockWorker(.15)
        self.safety_confirmation = {"note": "Per-cycle confirmation is required by the supervised extension."}

    def state(self, now):
        public = super().state(now)
        public.update(banner=self.banner, navigation=self.cycle.public(now))
        public["robot_commands_sent"] = (len(self.cycle.result["command_send_statuses"])
                                         if self.cycle.live and self.cycle.result else None if self.cycle.live else 0)
        return public

    def step(self, now):
        super().step(now)
        self.navigation.process(self.monotonic())

    def handle(self, line, typed_s, viewer, now):
        # Terminal is log/status + emergency request, not a second unbound target API.
        if line.strip() in ("quit", "cancel", "stop"):
            self.navigation.enqueue({"action": line.strip()}, now)
        elif line.strip() in ("status", "help"):
            self.say(json.dumps(self.cycle.public(now), indent=2))
        else:
            self.say("Use the supervised browser controls. Terminal: status | cancel | stop | quit.")

    def close(self, now):
        self.navigation.executor.cancel()
        closed = self.navigation.executor.close()
        if not closed:
            self.cycle.locked = True
            self.cycle.hold("executor_cleanup_unverified_independent_intervention_required")
        self.navigation.process(self.monotonic())
        (self.store.root / "navigation_summary.json").write_text(json.dumps(self.cycle.public(self.monotonic()), indent=2)+"\n")
        super().close(now)

    def execution_summary(self):
        return {"supervised_navigation": self.cycle.public(self.monotonic()),
                "robot_commands_sent": len(self.cycle.result["command_send_statuses"]) if self.cycle.live and self.cycle.result else None if self.cycle.live else 0,
                "physical_execution_authorized": self.cycle.live and self.cycle.result is not None}
