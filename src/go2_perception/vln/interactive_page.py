"""Single static page for the interactive review tool. Plain JS + canvas, no frameworks.

Panel B draws a native target point only on the exact model input image of the same
prediction (both come from one prediction record, swapped together after the image has
loaded), so an old point can never appear on a newer frame. Panel C draws all native
waypoints in their own model-local frame. Panel D is LiDAR only; nothing is registered
across panels. All text is inserted with textContent.
"""

REVIEW_STYLE = r"""
:root{--bg:#eef2f4;--panel:#fff;--ink:#142b39;--muted:#526875;--line:#d3dde2;--live:#0f7b4f;--replay:#1d5fa8;
--review:#7a3fa0;--warn:#a5480b;--stale:#8a6d00;--bad:#b3261e;--pt:#ff2fc8;--path:#127eae;--yaw:#b34c12}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#11181c;--panel:#1a2329;--ink:#e3ecf0;
--muted:#9cb0bb;--line:#2c3a42;--path:#4fb3e0;--yaw:#e48a4f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.4 system-ui,-apple-system,sans-serif}
header{display:flex;flex-wrap:wrap;gap:10px;align-items:center;padding:10px 16px;background:#173b50;color:#fff}
header strong{font-size:18px}.mode{font-weight:700;padding:4px 12px;border-radius:6px;font-size:16px;letter-spacing:.04em}
.LIVE{background:var(--live)}.REPLAY,.SYNTHETIC{background:var(--replay)}.REVIEW{background:var(--review)}
.strip{display:flex;flex-wrap:wrap;gap:6px;padding:8px 16px;background:var(--panel);border-bottom:1px solid var(--line)}
.chip{border:1px solid var(--line);border-radius:6px;padding:3px 8px;font-size:12px}.chip b{font-weight:600}
.chip.low{border-color:var(--bad);color:var(--bad)}
main{padding:12px 16px;max-width:1700px;margin:auto}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}@media(max-width:1000px){.grid{grid-template-columns:1fr}}
section{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px;min-width:0}
h2{font-size:15px;margin:0 0 6px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.badge{font-size:12px;font-weight:700;padding:2px 8px;border-radius:5px;color:#fff;background:var(--muted)}
.CURRENT{background:var(--live)}.STALE{background:var(--stale)}.SUPERSEDED{background:var(--muted)}.REJECTED{background:var(--bad)}
.INTERRUPTED{background:var(--bad)}.RECEIVING{background:var(--live)}
img.cam,canvas{width:100%;height:auto;display:block;background:#000;border-radius:6px}
canvas.chart{background:transparent;border:1px solid var(--line)}
.meta{font:12px ui-monospace,Menlo,monospace;color:var(--muted);white-space:pre-wrap;word-break:break-word;margin:6px 0 0}
.note{font-size:12px;color:var(--muted)}.warn{color:var(--warn);font-weight:600}
button{font:inherit;border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:6px;padding:3px 10px;cursor:pointer}
button.on{background:var(--replay);color:#fff}
table{font:12px ui-monospace,Menlo,monospace;border-collapse:collapse;width:100%}td,th{padding:2px 6px;text-align:right;border-bottom:1px solid var(--line)}
.sheet{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:10px}
.tile{border:1px solid var(--line);border-radius:8px;padding:6px}.tile canvas{cursor:crosshair}
#review{border:2px solid var(--review);margin-bottom:12px;display:none}
#test{border:3px solid var(--replay);margin-bottom:12px}
.steps{display:flex;gap:6px;margin:0 0 10px;flex-wrap:wrap}.steps span{padding:4px 12px;border-radius:999px;border:1px solid var(--line);font-size:13px}
.steps span.on{background:var(--replay);color:#fff;border-color:var(--replay)}.steps span.done{opacity:.55}
#tmsg{font-size:18px;font-weight:600;margin:4px 0 10px}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:6px 0}
input[type=text]{font:inherit;padding:6px 10px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--ink);min-width:220px}
button.big{font-size:16px;padding:8px 16px}button.primary{background:var(--replay);color:#fff;border-color:var(--replay)}
.choices button{font-size:15px;padding:8px 14px}.choices button.sel{background:var(--live);color:#fff}
#tanswer{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:12px}@media(max-width:900px){#tanswer{grid-template-columns:1fr}}
"""

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Go2 VLN Review</title>
<style>""" + REVIEW_STYLE + r"""</style></head><body>
<header><strong id="banner">ADVISORY ONLY — NO ROBOT COMMANDS</strong><span id="mode" class="mode">…</span>
<span id="trialhead"></span></header>
<div class="strip" id="strip"></div>
<main>
<section id="test"><h2>Test — one model answer at a time</h2>
<div class="steps"><span id="st1">1 · Choose target</span><span id="st2">2 · Model is answering</span><span id="st3">3 · Your verdict</span><span id="st4">4 · Saved</span></div>
<div id="tmsg"></div>
<div id="tready"><div class="row"><label for="ttarget">Target:</label><input id="ttarget" type="text" placeholder="e.g. chair" maxlength="200">
<button id="task" class="big primary">Ask the model</button><span style="flex:1"></span><button id="tquit">Close tool</button></div>
<div class="row" id="tsafety"><label><input type="checkbox" id="tsafe"> <span id="tsafetext"></span></label></div></div>
<div id="tasking" class="row"><span class="note" id="taskingtime"></span><button id="tcancel1">Cancel</button></div>
<div id="tanswerbox"><div id="tanswer"><canvas id="tcanvas" width="448" height="256"></canvas>
<div><div id="tverdict" style="font-size:18px;font-weight:700"></div><div id="tqmeta" class="meta"></div>
<p id="tquestion" style="font-weight:600"></p><div class="row choices" id="tchoices"></div>
<div class="row"><input id="tcomment" type="text" placeholder="Comment in your own words (optional)" maxlength="2000" style="flex:1"></div>
<div class="row"><button id="tsave" class="big primary">Save</button><button id="tcancel2">Discard</button></div></div></div></div>

<p class="note">Read-only: the model only proposes; nothing is sent to the robot. Keep the robot still while the model answers. Your verdicts are stored separately and never given to the model. Panels A–D below are live context.</p></section>
<section id="review"><h2><span class="badge" style="background:var(--review)">REVIEW</span><span id="rtitle"></span></h2>
<p class="note">Saved records, separated by origin: model outputs (never labels), operator marks/notes (verbatim), confirmed human reviews, and optional target corrections. Click a prediction image to store a correction for that image only (separate file, never sent to the model). Answer the review questions in the terminal.</p>
<div id="rmeta" class="meta"></div><div class="sheet" id="sheet"></div><div id="rlabels" class="meta"></div></section>
<div class="grid">
<section><h2>A · Live camera <span id="camstate" class="badge"></span></h2><img id="cam" class="cam" alt="Latest camera frame">
<div id="cammeta" class="meta"></div></section>
<section><h2>B · Raw model mark <span id="plabel" class="badge"></span><span id="pid" class="note"></span></h2>
<canvas id="pimg" width="448" height="256"></canvas><div id="pmeta" class="meta"></div>
<p class="note">The pink cross is an unverified model proposal on this exact input. CURRENT describes timing. Object identification is shown separately.</p></section>
<section><h2>C · Proposed waypoints and headings <span id="clabel" class="badge"></span></h2>
<canvas id="path" class="chart" width="440" height="330"></canvas><table id="wtab"></table><div id="cmeta" class="meta"></div>
<p class="note">Model-local SE(2) frame of the decision image (x forward, y left, yaw CCW; metres/radians). No time base or speeds. Not registered to LiDAR, base or camera: no measured transform.</p></section>
<section><h2>D · LiDAR top-down (base_link) <span id="lstate" class="badge"></span></h2>
<div><button data-v="latest">latest scan</button> <button data-v="0.5">accumulated 0.5 s</button> <button data-v="1.0">accumulated 1.0 s</button></div>
<canvas id="lidar" class="chart" width="440" height="440"></canvas><div id="lmeta" class="meta"></div></section>
</div></main>
<script>
const TOKEN="__TOKEN__";let S=null,shownFrame=null,shownPred=null,shownPredObj=null,lidarView="latest",lastReport=0,loadingPred=null;
const $=id=>document.getElementById(id);
const fmt=(v,d=3)=>v===null||v===undefined?"UNKNOWN":(typeof v==="number"?v.toFixed(d):String(v));
const gib=b=>b==null?"?":(b/2**30).toFixed(2)+" GiB";
function chip(k,v,low){const c=document.createElement("span");c.className="chip"+(low?" low":"");const b=document.createElement("b");b.textContent=k+": ";c.append(b,document.createTextNode(v));return c}
function report(){lastReport=Date.now();fetch("/api/viewing",{method:"POST",headers:{"Content-Type":"application/json","X-Review-Token":TOKEN},
 body:JSON.stringify({prediction_id:shownPred,camera_frame_id:shownFrame,lidar_view:lidarView})}).catch(()=>{})}
function strip(){const s=$("strip");s.replaceChildren();const r=S.resources||{},m=S.model||{},c=S.camera||{};const rss=r.rss_bytes_by_process||{};
 s.append(chip("trial",S.trial?S.trial.trial_id+" ("+fmt(S.trial.elapsed_s,1)+" s)":"none"),
 chip("model",(m.state||"?")+(m.pending?" +pending":"")),chip("inference",m.last_inference_s==null?"—":fmt(m.last_inference_s,2)+" s"),
 chip("camera age",c.host_since_receipt_s==null?"UNKNOWN":fmt(c.host_since_receipt_s,2)+" s since receipt"+(c.since_estimated_exposure_s!=null?" / "+fmt(c.since_estimated_exposure_s,2)+" s est. exposure":"")),
 chip("recording",S.recording&&S.recording.trial?"ON "+S.recording.trial:"off"),
 chip("memory","app "+gib(rss.app_parent)+" · worker "+(rss.model_worker==null?"n/a":gib(rss.model_worker))+" · avail "+gib(r.system_available_bytes),(r.low||[]).some(x=>x.includes("memory")||x.includes("RSS"))),
 chip("queue","cam inbox "+(c.inbox_queued??"?")+"/"+(c.inbox_capacity??"?")+" · overflow "+((c.counts||{}).inbox_overflow??0)+(S.lidar&&S.lidar.recording?" · lidar writer "+S.lidar.recording.queued+"/"+S.lidar.recording.queue_max:"")),
 chip("disk","run "+gib(r.run_bytes)+" · free "+gib(r.disk_free_bytes),(r.low||[]).some(x=>x.includes("disk")||x.includes("quota"))));
 if(S.resource_stop)s.append(chip("STOPPED",S.resource_stop.join("; "),true))}
function panelA(){const c=S.camera;if(!c){$("camstate").textContent="NO CAMERA";$("cammeta").textContent="Camera source not configured.";return}
 $("camstate").textContent=c.state;$("camstate").className="badge "+c.state.split(" ")[0];const L=c.latest;
 if(L&&L.frame_id!==shownFrame){const id=L.frame_id,img=new Image();img.onload=()=>{$("cam").src=img.src;shownFrame=id;report()};img.src="/api/camera?f="+encodeURIComponent(id)}
 if(!L){$("cammeta").textContent="Waiting for the first frame.";return}
 $("cammeta").textContent=["frame "+L.frame_id,"receipt (host "+L.stamp_host.clock+", "+L.stamp_host.semantics+"): "+fmt(L.stamp_host.t,3),
 "robot-clock stamp: "+(L.stamp_robot?fmt(L.stamp_robot.t,3)+" ("+L.stamp_robot.clock+", "+L.stamp_robot.semantics+")":"NONE (not subscribed for RGB; never relabelled)"),
 "clock epoch: "+L.clock_epoch,"fps (receipt, 2 s): "+fmt(c.fps_receipt_2s,1)+" · "+(L.rectified?"RECTIFIED":"RAW (distorted, not rectified)")+" · "+L.shape.join("×"),
 "receipt gaps > 0.375 s: "+c.counts.receipt_gaps+(c.last_gap?" (last "+fmt(c.last_gap.after_s,2)+" s before "+c.last_gap.at_frame+")":""),
 "estimated exposure: "+(L.exposure_monotonic_s==null?"UNKNOWN":"configured latency model (unvalidated on this boot)")].join("\n")}
function drawPred(p,img){const cv=$("pimg"),g=cv.getContext("2d");cv.width=img?img.naturalWidth:448;cv.height=img?img.naturalHeight:256;
 g.clearRect(0,0,cv.width,cv.height);if(img)g.drawImage(img,0,0);else{g.fillStyle="#333";g.fillRect(0,0,cv.width,cv.height)}
 if(img&&p.target_fraction){const x=p.target_fraction[0]*cv.width,y=p.target_fraction[1]*cv.height;g.strokeStyle="#000";g.lineWidth=6;
  g.beginPath();g.moveTo(x-14,y);g.lineTo(x+14,y);g.moveTo(x,y-14);g.lineTo(x,y+14);g.stroke();g.strokeStyle=getComputedStyle(document.body).getPropertyValue("--pt");g.lineWidth=3;g.stroke()}
 if(p.opos_not_visible){g.fillStyle="rgba(0,0,0,.65)";g.fillRect(0,0,cv.width,34);g.fillStyle="#fff";g.font="bold 18px system-ui";g.fillText("OPOS-0: model says target NOT VISIBLE",10,23)}
 shownPred=p.prediction_id;shownPredObj=p;report();drawPath(p)}
function panelB(){const p=S.prediction;if(!p){loadingPred=null;
 if(!S.navigation?.last_observation||S.navigation.state!=="TERMINAL"){
   const cv=$("pimg");cv.getContext("2d").clearRect(0,0,cv.width,cv.height);shownPred=null;}$("plabel").textContent="NO PREDICTION";$("pid").textContent="";$("pmeta").textContent="No model output yet. Use the Test box above.";shownPredObj=null;drawPath(null);return}
 if(p.prediction_id!==shownPred&&p.prediction_id!==loadingPred){loadingPred=p.prediction_id;const snap=p;
  const cv=$("pimg");cv.getContext("2d").clearRect(0,0,cv.width,cv.height);shownPred=null;shownPredObj=null;
  $("pid").textContent="";$("pmeta").textContent="Loading this prediction's source image";$("plabel").textContent="PENDING";
  const finish=img=>{if(loadingPred!==snap.prediction_id||S.prediction?.prediction_id!==snap.prediction_id)return;
    loadingPred=null;drawPred(snap,img)};
  if(p.image_url){const img=new Image();img.onload=()=>finish(img);img.onerror=()=>finish(null);img.src=p.image_url+"&t="+Date.now()}
  else{finish(null)}}
 if(p.prediction_id!==shownPred)return;const t=p.display_timing||{};$("plabel").textContent=p.label;$("plabel").className="badge "+p.label;$("clabel").textContent=p.label;$("clabel").className="badge "+p.label;$("pid").textContent=p.prediction_id+(p.mock?" · MOCK — no model":"");
 $("pmeta").textContent=["trial "+p.trial_id+(S.trial&&S.trial.trial_id===p.trial_id?" (active)":" (not the active trial)"),"instruction: "+p.instruction,
 p.opos_not_visible?"native target: OPOS-0 = NOT VISIBLE (model claim, not proof of absence)":p.target_fraction?"native target point (model claim): "+p.target_fraction.map(v=>v.toFixed(3)).join(", ")+" of image":"native target: none decodable",
 "source frame "+p.source_frame.frame_id+" · t="+fmt(p.source_frame.observation_s,3)+" ("+p.source_frame.clock+", "+p.source_frame.semantics+")",
 "age at display: "+fmt(t.observation_age_at_display_s,2)+" s ["+(t.age_basis||"")+"] · host since delivery "+fmt(t.host_since_delivery_s,2)+" s",
 "status reason: "+(p.current_rejection_reason||"none (within the unchanged advisory limit)"),
 "inference "+fmt(p.inference_duration_s,2)+" s · native generate "+fmt(p.native_generate_s,2)+" s · "+p.history_frames_in_job+" history frames · "+(p.mock?"MOCK (no model)":p.backend),
 "image: "+(p.source_image?p.source_image.kind:"unavailable"),"native: "+(p.native_output||"")].join("\n")}
function drawPath(p){const cv=$("path"),g=cv.getContext("2d");g.clearRect(0,0,cv.width,cv.height);const ink=getComputedStyle(document.body).getPropertyValue("--muted");
 g.strokeStyle=ink;g.lineWidth=1;g.beginPath();g.moveTo(220,320);g.lineTo(220,10);g.moveTo(10,250);g.lineTo(430,250);g.stroke();g.fillStyle=ink;g.font="11px system-ui";g.fillText("+x forward",226,20);g.fillText("+y left",14,242);
 $("clabel").textContent=p?p.label:"";$("clabel").className="badge "+(p?p.label:"");const tab=$("wtab");tab.replaceChildren();
 if(!p||!Array.isArray(p.waypoints)||p.waypoints.length===0){$("cmeta").textContent="No native waypoint chunk.";return}
 const w=p.waypoints,m=Math.max(0.25,...w.map(r=>Math.max(Math.abs(r[0]),Math.abs(r[1])))),k=200/m;g.fillText("scale: "+m.toFixed(2)+" m to edge",300,320);
 const pts=w.map(r=>[220-r[1]*k,250-r[0]*k]);g.strokeStyle=getComputedStyle(document.body).getPropertyValue("--path");g.lineWidth=2;g.beginPath();g.moveTo(220,250);pts.forEach(q=>g.lineTo(q[0],q[1]));g.stroke();
 pts.forEach((q,i)=>{const yaw=w[i][2];g.fillStyle=getComputedStyle(document.body).getPropertyValue("--path");g.beginPath();g.arc(q[0],q[1],3,0,7);g.fill();
  g.strokeStyle=getComputedStyle(document.body).getPropertyValue("--yaw");g.beginPath();g.moveTo(q[0],q[1]);g.lineTo(q[0]-Math.sin(yaw)*14,q[1]-Math.cos(yaw)*14);g.stroke();g.fillStyle=ink;g.fillText(String(i),q[0]+5,q[1]-4)});
 g.fillStyle="#111";g.beginPath();g.arc(220,250,5,0,7);g.fill();
 const h=tab.insertRow();["#","x fwd m","y left m","yaw CCW rad"].forEach(t=>{const c=document.createElement("th");c.textContent=t;h.append(c)});
 w.forEach((r,i)=>{const row=tab.insertRow();[i,...r].forEach((v,j)=>{row.insertCell().textContent=j===0?String(v):Number(v).toFixed(3)})});
 $("cmeta").textContent=["prediction "+p.prediction_id+" (same as panel B)","native STOP: "+fmt(p.native_stop)+" · explicit STOP code: "+fmt(p.explicit_stop_code),
 "model completion claim: "+fmt(p.model_completion_claim)+" · verified completion: UNKNOWN","execution: "+p.execution_status,"convention: "+p.waypoint_convention+" · time base: NONE"].join("\n")}
let lidarPts=null;
async function pollLidar(){if(S&&S.lidar){try{const r=await fetch("/api/lidar?view="+lidarView);lidarPts=r.ok?(await r.json()).points:null}catch(e){lidarPts=null}}drawLidar();setTimeout(pollLidar,300)}
function drawLidar(){const cv=$("lidar"),g=cv.getContext("2d");g.clearRect(0,0,440,440);const L=S&&S.lidar;
 document.querySelectorAll("button[data-v]").forEach(b=>b.classList.toggle("on",b.dataset.v===lidarView));
 if(!L){$("lstate").textContent="NOT CONNECTED";$("lmeta").textContent="No LiDAR source configured: geometry UNKNOWN (absence of points is not free space).";return}
 const st=L.state,k=40,ox=220,oy=300;const ink=getComputedStyle(document.body).getPropertyValue("--muted");
 g.strokeStyle=ink;g.lineWidth=.5;for(let r=1;r<=7;r++){g.beginPath();g.arc(ox,oy,r*k,0,7);g.stroke()}g.fillStyle=ink;g.font="10px system-ui";for(let r=1;r<=7;r++)g.fillText(r+" m",ox+3,oy-r*k+10);
 g.fillStyle="rgba(150,150,150,.35)";g.beginPath();g.arc(ox,oy,0.3*k,0,7);g.fill();
 if(lidarPts)lidarPts.forEach(p=>{const x=ox-p[1]*k,y=oy-p[0]*k;if(x<0||x>440||y<0||y>440)return;const z=Math.max(0,Math.min(1,(p[2]+0.4)/1.6));
  g.fillStyle="hsl("+(220-200*z)+",80%,50%)";g.fillRect(x-1,y-1,2,2)});
 g.fillStyle="#111";g.beginPath();g.moveTo(ox,oy-9);g.lineTo(ox-6,oy+6);g.lineTo(ox+6,oy+6);g.fill();
 const state=!L.bridge_alive?(L.ended?"ENDED":"DOWN"):st&&st.latest_scan&&st.latest_scan.host_age_s<0.5?"RECEIVING":"STALE";$("lstate").textContent=(L.source||"")+" · "+state;
 $("lstate").className="badge "+(state==="RECEIVING"?"RECEIVING":"INTERRUPTED");
 if(!st){$("lmeta").textContent=L.error?"error: "+L.error:"Waiting for LiDAR data.";return}
 const win=st.windows[lidarView],ls=st.latest_scan||{},cov=(st.coverage||{})[lidarView],sync=st.sync,pose=st.pose;
 $("lmeta").textContent=["view: "+lidarView+(lidarView==="latest"?" (single cloud_base scan)":win&&win.available?" · "+win.frames+" scans over "+fmt(win.span_robot_s,2)+" s (odom-compensated)":" · UNAVAILABLE: "+(win?win.reason:"")),
 "latest scan #"+fmt(ls.seq,0)+" · "+fmt(ls.n,0)+" pts · host age "+fmt(ls.host_age_s,2)+" s · robot t_mid "+fmt(ls.t_mid,3),
 "intervals (robot) median "+fmt(st.robot_scan_intervals&&st.robot_scan_intervals.median_s)+" max "+fmt(st.robot_scan_intervals&&st.robot_scan_intervals.max_s)+" s · (host receipt) median "+fmt(st.host_receipt_intervals&&st.host_receipt_intervals.median_s)+" max "+fmt(st.host_receipt_intervals&&st.host_receipt_intervals.max_s)+" s",
 "coverage: "+(cov?cov.sectors_with_returns+"/"+cov.sectors+" azimuth sectors, "+(100*cov.grid_fraction).toFixed(1)+"% of 10×10 m cells have returns; the rest UNKNOWN":"UNKNOWN"),
 "pose: "+(pose.valid?"x "+fmt(pose.x,2)+" y "+fmt(pose.y,2)+" yaw "+fmt(pose.yaw_rad,3)+" ("+pose.frame+")":"UNKNOWN — "+pose.reason),
 "robot clock epoch: "+st.robot_epoch+" · sync: "+sync.status.toUpperCase()+(sync.status==="estimated"?" host−robot ≈ "+fmt(sync.host_minus_robot_s,3)+" s (estimate)":" — "+sync.reason+"; streams shown separately"),
 "counts: scan gaps "+st.counts.scan_seq_gaps+" (missing "+st.counts.scans_missing+") · epoch changes "+st.counts.robot_epoch_changes+" · refused (no pose) "+JSON.stringify(st.counts.refused_no_pose),
 ...st.unknowns.map(u=>"unknown: "+u)].join("\n")}
document.querySelectorAll("button[data-v]").forEach(b=>b.onclick=()=>{lidarView=b.dataset.v;report()});
let tChoice=null,tShown=null,tLoaded=null,tSaving=false,tAnswer=null;
function act(body){return fetch("/api/guided",{method:"POST",headers:{"Content-Type":"application/json","X-Review-Token":TOKEN},body:JSON.stringify(body)})}
$("task").onclick=()=>{act({action:"ask",target:$("ttarget").value,safety_confirmed:$("tsafe").checked})};
$("ttarget").addEventListener("keydown",e=>{if(e.key==="Enter")$("task").click()});
$("tcancel1").onclick=$("tcancel2").onclick=()=>act({action:"cancel"});
$("tquit").onclick=()=>{if(confirm("Close the tool? Saved tests stay on disk. Nothing is sent to the robot."))act({action:"quit"})};
function canSave(G){return G&&G.step==="ANSWER"&&tAnswer&&tShown===tAnswer.prediction_id&&tLoaded===tShown&&tChoice&&!tSaving&&!G.pending_save}
$("tsave").onclick=async()=>{const G=S&&S.guided;if(!tChoice){alert("Choose one of the verdict buttons first.");return}
 if(!canSave(G))return;tSaving=true;$("tsave").disabled=true;
 try{const r=await act({action:"answer",choice:tChoice,comment:$("tcomment").value,trial_id:tAnswer.trial_id,
   prediction_id:tAnswer.prediction_id,image_sha256:tAnswer.image_sha256});
  if(!r.ok){const e=await r.json().catch(()=>({}));alert("Not saved: "+(e.error||r.status))}}finally{tSaving=false}};
function tCover(text){const cv=$("tcanvas"),g=cv.getContext("2d");cv.width=448;cv.height=256;g.fillStyle="#333";g.fillRect(0,0,448,256);
 g.fillStyle="#fff";g.font="16px system-ui";g.fillText(text,16,130)}
function panelTest(){const G=S.guided;$("test").style.display=S.mode==="REVIEW"?"none":"block";if(!G||S.mode==="REVIEW")return;const order=["READY","ASKING","ANSWER","SAVED"],i=order.indexOf(G.step);
 ["st1","st2","st3","st4"].forEach((id,k)=>{$(id).className=k===i?"on":k<i?"done":""});
 $("tmsg").textContent=G.message;$("tready").style.display=(G.step==="READY"||G.step==="SAVED")?"block":"none";
 $("tsafety").style.display=G.safety_needed?"flex":"none";$("tsafetext").textContent="I confirm: "+G.safety_statement;
 $("tasking").style.display=G.step==="ASKING"?"flex":"none";$("taskingtime").textContent=G.step==="ASKING"?"Asking the model about “"+G.target+"” — "+fmt(G.asking_for_s,0)+" s (usually 2–4 s)":"";
 if(G.step==="SAVED"&&tShown){$("ttarget").value="";tShown=null}$("tanswerbox").style.display=G.step==="ANSWER"?"block":"none";
 if(G.step!=="ANSWER"){tShown=null;tChoice=null;tLoaded=null;tAnswer=null;return}
 const a=G.answer;if(!a)return;
 if(tShown!==a.prediction_id){const id=a.prediction_id;tShown=id;tAnswer=a;tChoice=null;tLoaded=null;$("tcomment").value="";
  tCover("Loading the model's image for "+id.split("-").pop()+" …");  // never show the previous image
  if(a.image_url){const img=new Image();
   img.onload=()=>{if(tShown!==id)return;  // a late callback from an older answer
    const cv=$("tcanvas"),g=cv.getContext("2d");cv.width=img.naturalWidth;cv.height=img.naturalHeight;g.drawImage(img,0,0);
    if(a.target_fraction){const x=a.target_fraction[0]*cv.width,y=a.target_fraction[1]*cv.height;g.strokeStyle="#000";g.lineWidth=7;g.beginPath();g.moveTo(x-16,y);g.lineTo(x+16,y);g.moveTo(x,y-16);g.lineTo(x,y+16);g.stroke();g.strokeStyle="#ff2fc8";g.lineWidth=4;g.stroke()}
    tLoaded=id};
   img.onerror=()=>{if(tShown!==id)return;tCover("The model's image failed to load. Discard and ask again.")};
   img.src=a.image_url+"&t="+Date.now()}
  else tCover("No archived image for this answer. Discard and ask again.");
  const box=$("tchoices");box.replaceChildren();Object.entries(G.choices).forEach(([k,v])=>{const b=document.createElement("button");b.textContent=v;b.onclick=()=>{tChoice=k;[...box.children].forEach(c=>c.classList.toggle("sel",c===b))};box.append(b)})}
 $("tsave").disabled=!canSave(G);$("tsave").textContent=G.pending_save||tSaving?"Saving…":tLoaded!==tShown?"Waiting for image…":"Save";
 $("tverdict").textContent=a.opos_not_visible?"Model: the "+G.target+" is NOT VISIBLE here.":a.target_fraction?"Unverified model mark for "+G.target+": pink + .":"Model gave no usable point.";
 $("tquestion").textContent=a.target_fraction?"Is the + on the "+G.target+"?":a.opos_not_visible?"Is that right — is the "+G.target+" really not in view?":"What do you see?";
 $("tqmeta").textContent=["answer "+a.prediction_id.split("-").pop()+" · model took "+fmt(a.inference_duration_s,1)+" s"+(a.mock?" · MOCK (no model)":""),
  "status when the answer arrived: "+a.label_at_receipt+" (observation was "+fmt(a.age_at_receipt_s,1)+" s old since receipt"+(a.exposure_age_at_receipt_s!=null?", "+fmt(a.exposure_age_at_receipt_s,1)+" s since estimated exposure":"")+")",
  "this image is now "+fmt(a.age_now_s,0)+" s old: judge THIS image, not the live view",
  "image = exactly what the model saw (frame "+a.source_frame_id+")"].join("\n")}
function reviewEvidence(e){
 if(e.rejection)return "Executor held this observation · "+e.rejection;
 const g=e.grounding, candidates=e.reviews.map(r=>r.decision).join(", ");
 const support=g.status==="supported"?(g.method==="synthetic_fixture"?"Synthetic fixture support":"Matched named reference"):
   g.status==="absent"?"Model reports target absent":g.status==="uncertain"?"Target uncertain":"Unverified candidate";
 const sel=e.selection, motion=sel?sel.mode+" · "+sel.source+" · "+sel.action.direction:"No dispatch recorded for this prediction";
 return [support,g.method,candidates?"Image review: "+candidates:"",motion].filter(Boolean).join(" · ")}
let reviewKey=null;
function panelReview(){const R=S.review,box=$("review");if(!R){box.style.display="none";reviewKey=null;return}box.style.display="block";
 $("rtitle").textContent=R.trial_id+" · "+R.trial.instruction+(R.dialog_open?" · answer the questions in the terminal":"");
 $("rmeta").textContent=["operator text (request, not a verified label): "+R.trial.operator_text+(R.trial.template_id?" · template "+R.trial.template_id:""),
 "ended: "+(R.end?R.end.end_reason+" at "+R.end.ended_wall_utc:"not ended"),"predictions "+R.predictions.length+" · marks "+R.marks.length+" · notes "+R.notes.length+" · confirmed human reviews "+R.human_labels.length+" · corrections "+R.corrections.length].join("\n");
 const key=R.trial_id+":"+R.predictions.length+":"+R.marks.length+":"+R.corrections.length;
 if(key!==reviewKey){reviewKey=key;const sh=$("sheet");sh.replaceChildren();
  R.predictions.forEach(p=>{const t=document.createElement("div");t.className="tile";const cv=document.createElement("canvas");const m=document.createElement("div");m.className="meta";
   m.textContent=[p.prediction_id+" · "+p.label_at_receipt+(p.rejection_reason_at_receipt?" ("+p.rejection_reason_at_receipt+")":""),p.opos_not_visible?"OPOS-0: model reports absence":p.target_fraction?"Unverified model mark "+p.target_fraction.map(v=>v.toFixed(3)).join(", "):"no point",
    p.episode_review?reviewEvidence(p.episode_review):"Independent instance support unavailable for this image",
    "STOP "+p.native_stop+" · src "+p.source_frame.frame_id,"src t="+fmt(p.source_frame.observation_s,3)+" ("+p.source_frame.clock+")"].join("\n");
   t.append(cv,m);sh.append(t);const img=new Image();img.onload=()=>{cv.width=img.naturalWidth;cv.height=img.naturalHeight;const g=cv.getContext("2d");g.drawImage(img,0,0);
    if(p.target_fraction){const x=p.target_fraction[0]*cv.width,y=p.target_fraction[1]*cv.height;g.strokeStyle="#ff2fc8";g.lineWidth=3;g.beginPath();g.moveTo(x-12,y);g.lineTo(x+12,y);g.moveTo(x,y-12);g.lineTo(x,y+12);g.stroke()}
    R.corrections.filter(c=>c.prediction_id===p.prediction_id).forEach(c=>{g.strokeStyle="#00d0ff";g.lineWidth=3;g.beginPath();g.arc(c.x_fraction*cv.width,c.y_fraction*cv.height,9,0,7);g.stroke()})};
   img.src="/api/pred_image?id="+encodeURIComponent(p.prediction_id);
   cv.onclick=e=>{const r=cv.getBoundingClientRect();const x=(e.clientX-r.left)/r.width,y=(e.clientY-r.top)/r.height;
    if(!confirm("Store a target correction at ("+x.toFixed(3)+", "+y.toFixed(3)+") for "+p.prediction_id+"? It is saved separately and never sent to the model."))return;
    fetch("/api/correction",{method:"POST",headers:{"Content-Type":"application/json","X-Review-Token":TOKEN},body:JSON.stringify({prediction_id:p.prediction_id,x_fraction:x,y_fraction:y,natural_width:cv.width,natural_height:cv.height})})}});
  R.marks.forEach(mk=>{const t=document.createElement("div");t.className="tile";const im=document.createElement("img");im.className="cam";im.alt="Marked camera frame";
   if(mk.camera_frame&&mk.camera_frame.image)im.src="/api/mark_image?trial="+encodeURIComponent(R.trial_id)+"&mark="+encodeURIComponent(mk.mark_id);
   const m=document.createElement("div");m.className="meta";m.textContent=["MARK "+mk.mark_id+" ("+mk.binding+")","prediction "+(mk.prediction?mk.prediction.prediction_id:"none shown")+" · label then: "+fmt(mk.prediction_label_when_marked),
   "frame "+(mk.camera_frame?mk.camera_frame.frame_id:"none"),"note (verbatim): "+(mk.verbatim_note||"")].join("\n");t.append(im,m);sh.append(t)})}
 $("rlabels").textContent=[...R.notes.map(n=>"NOTE "+n.note_id+" ("+n.context+"): "+n.verbatim_note),...R.human_labels.map(h=>h.schema==="go2-vln-guided-answer-review-v1"?"GUIDED VERDICT on "+h.prediction_id+" by "+(h.reviewer||"?")+" at "+h.reviewed_wall_utc+": "+h.choice_text+" ("+h.choice+") · comment (verbatim): "+(h.verbatim_comment||"—"):"HUMAN REVIEW "+h.review_id+" by "+h.reviewer+" at "+h.reviewed_wall_utc+": present="+h.target_presence+" marking="+h.marking+" path="+h.path+" stopping="+h.stopping+" · comment (verbatim): "+h.verbatim_comment)].join("\n")||"No notes or confirmed human reviews yet."}
async function poll(){try{const r=await fetch("/api/state");S=await r.json();$("mode").textContent=S.view;$("mode").className="mode "+S.view;
 $("trialhead").textContent=S.trial?"trial "+S.trial.trial_id+" · "+S.trial.instruction+(S.trial.feeding_paused&&S.trial.feeding_paused!=="single_answer_requested"?" · INFERENCE PAUSED: "+S.trial.feeding_paused:""):"no active trial";
 strip();panelTest();panelReview();panelA();panelB();if(!S.prediction)drawPath(null);if(Date.now()-lastReport>1000)report()}catch(e){$("mode").textContent="DISCONNECTED";$("mode").className="mode"}setTimeout(poll,250)}
poll();pollLidar();
</script></body></html>"""
