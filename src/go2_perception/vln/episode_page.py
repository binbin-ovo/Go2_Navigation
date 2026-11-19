"""Add episode controls to the existing camera, native image, trajectory and LiDAR viewer."""

from .interactive_page import PAGE

CONTROLS = """
<style>#test{display:none!important}#episode button{margin:.2em}#estop{background:#97272f;color:white}
#pimg{touch-action:none}#etrace{white-space:pre-wrap}</style>
<section id="episode"><h2>Supervised search and approach</h2>
<p id="ebounds"></p><input id="eobject" maxlength="160" aria-label="Requested object description" placeholder="brown cardboard box on the floor">
<input id="eoperator" maxlength="100" placeholder="Operator name">
<label><input type="checkbox" id="earea">I inspected the entire declared area, blind rear, stopping room and cable.</label>
<label><input type="checkbox" id="esource">I have exclusive command access and an independent intervention control.</label>
<button id="earm">Authorize episode</button><button id="estop">STOP EPISODE</button>
<p id="estatus" role="status"></p><div id="eidentify" hidden>
For Identify or Reject, first drag a tight rectangle around the candidate on the exact image in panel B. Unsure requests a new view.
<button id="eyes">This is the requested target</button><button id="eno">This is a different object</button><button id="euncertain">Unsure · seek another view</button></div>
<button id="egoal" hidden>I confirm arrival on this image</button><pre id="etrace"></pre></section>
"""

SCRIPT = """
let episodeImage=null, episodeROI=null, dragStart=null;
function coords(ev){const r=$("pimg").getBoundingClientRect();
  return [Math.max(0,Math.min(1,(ev.clientX-r.left)/r.width)),Math.max(0,Math.min(1,(ev.clientY-r.top)/r.height))]}
$("pimg").addEventListener("pointerdown",ev=>{if(!episodeImage||S.navigation.state!=="IDENTIFYING")return;
  dragStart=coords(ev);$("pimg").setPointerCapture(ev.pointerId)});
$("pimg").addEventListener("pointerup",ev=>{if(!dragStart)return;const [a,b]=dragStart,[x,y]=coords(ev);
  episodeROI=[Math.min(a,x),Math.min(b,y),Math.max(a,x),Math.max(b,y)];dragStart=null;
  const cv=$("pimg"),g=cv.getContext("2d");g.strokeStyle="#ee8900";g.lineWidth=3;
  g.strokeRect(a*cv.width,b*cv.height,(x-a)*cv.width,(y-b)*cv.height)});
async function sendEpisode(action){const p={action,operator:$("eoperator").value.trim(),
  target:$("eobject").value,area_inspected:$("earea").checked,exclusive_and_intervention_confirmed:$("esource").checked};
  if(["identify","reject","uncertain","goal"].includes(action)){if(!episodeImage)return;Object.assign(p,episodeImage,{roi:episodeROI})}
  const r=await fetch("/api/navigation",{method:"POST",headers:{"Content-Type":"application/json","X-Review-Token":TOKEN},body:JSON.stringify(p)});
  if(!r.ok)$("estatus").textContent="Request rejected ("+r.status+")"}
for(const [id,act] of [["earm","authorize"],["estop","stop"],["eyes","identify"],["eno","reject"],["euncertain","uncertain"],["egoal","goal"]])
  $(id).onclick=()=>sendEpisode(act);
function overlayEpisode(e){
  const last=e.last_observation;
  const p=S.prediction || (last&&{...last,label:"ARCHIVED",waypoint_convention:"absolute ego SE(2), metres and radians",
    execution_status:"See episode attempts below"});
  if(!p)return;
  const s=e.selected;
  drawPath(p);
  if(!s||!s.action||s.prediction_id!==p.prediction_id)return;
  const cv=$("path"),g=cv.getContext("2d"),a=s.action;
  const m=Math.max(.25,...p.waypoints.map(r=>Math.max(Math.abs(r[0]),Math.abs(r[1])))),k=200/m;
  g.save();g.strokeStyle="#e06d0e";g.lineWidth=4;g.setLineDash([6,3]);g.beginPath();
  if(a.type==="forward"){g.moveTo(220,250);g.lineTo(220,250-k*a.magnitude)}
  else{const angle=(a.direction==="left"?1:-1)*a.magnitude;
    g.arc(220,250,48,-Math.PI/2,-Math.PI/2-angle,angle>0);g.moveTo(220,250);g.lineTo(220-48*Math.sin(angle),250-48*Math.cos(angle))}
  g.stroke();g.restore();$("cmeta").textContent+=" · Orange dashed: "+s.mode+" · "+s.source+" · selected "+a.direction+" "+a.magnitude.toFixed(3)+" "+a.unit}
function episodePanel(){
  const e=S.navigation;if(!e)return;$("banner").textContent=S.banner;const cfg=e.settings,last=e.last_observation, terminal=e.state==="TERMINAL";
  $("ebounds").textContent="One authorization · "+cfg.max_moves+" moves · "+cfg.seconds+" s · inspected radius "+cfg.area_radius_m+" m · approach band ±"+cfg.alignment_deg+"°";
  $("earm").disabled=e.state!=="IDLE";$("estatus").textContent=[e.target,e.mode,e.moves+"/"+e.max_moves,e.state,e.outcome,e.reason,e.ui_error].filter(Boolean).join(" · ");
  $("eidentify").hidden=e.state!=="IDENTIFYING";
  if(terminal&&last&&(!episodeImage||episodeImage.prediction_id!==last.prediction_id)){
    const id=last.prediction_id;
    if(window.episodeEndImage!==id){window.episodeEndImage=id;const im=new Image();im.onload=()=>{
      if(S.navigation.last_observation.prediction_id!==id)return;const cv=$("pimg"),g=cv.getContext("2d");
      cv.width=im.naturalWidth;cv.height=im.naturalHeight;g.drawImage(im,0,0);
      if(last.point_fraction){const x=last.point_fraction[0]*cv.width,y=last.point_fraction[1]*cv.height;
        g.strokeStyle="#ff2fc8";g.lineWidth=3;g.strokeRect(x-6,y-6,12,12)}
      episodeImage={prediction_id:id,image_sha256:last.image_sha256};$("pid").textContent=id};
    im.src="/api/pred_image?id="+encodeURIComponent(id)}}
  const bound=episodeImage&&last&&episodeImage.prediction_id===last.prediction_id&&episodeImage.image_sha256===last.image_sha256
    &&last.grounding?.image_sha256===episodeImage.image_sha256&&last.grounding?.prediction_id===last.prediction_id
    &&last.grounding?.target===e.target&&(terminal||S.prediction?.prediction_id===episodeImage.prediction_id);
  const g=bound?last.grounding:null;
  const status=!g?"Awaiting verification on this image":g.status==="supported"?
    (g.method==="synthetic_fixture"?"Synthetic fixture support":"Matched the identified reference on this image"):
    g.status==="absent"?"Model reports target absent":g.status==="uncertain"?"Uncertain · "+g.method:"Unverified model mark · identify or reject";
  $("groundStatus").textContent=(e.target?"Requested: "+e.target+" · ":"")+status;
  $("eyes").disabled=$("eno").disabled=!bound||!episodeROI;
  $("euncertain").disabled=!bound;
  $("egoal").hidden=!e.arrival_review_available;
  const age=last?Math.max(0,S.host_monotonic_s-last.observed_monotonic_s):null;
  const t=last?["Image "+last.prediction_id+" · source "+last.observed_monotonic_s.toFixed(3)+" s · age now "+age.toFixed(2)+" s · decision age "+last.age_at_decision_s.toFixed(2)+" s",
    "Visibility "+last.grounding.status+" · native STOP "+last.native_stop]:[];
  for(const a of (e.attempts||[]))t.push("#"+a.slot+" "+a.mode+" "+a.source+" "+a.direction+" requested "+a.requested.toFixed(3)+" "+a.unit+
    " · settled "+(a.measured==null?"pending":a.measured.toFixed(3))+" · "+(a.outcome||a.status||"in progress"));
  $("etrace").textContent=t.join("\\n");overlayEpisode(e)}
"""


def episode_page():
    page = PAGE.replace('<section id="test">', CONTROLS + '<section id="test">', 1)
    page = page.replace('<canvas id="pimg"', '<p id="groundStatus" role="status">Awaiting verification on this image</p><canvas id="pimg"', 1)
    page = page.replace("ADVISORY ONLY — NO ROBOT COMMANDS", "SUPERVISED EPISODE · CHECK MODE")
    page = page.replace(
        "function drawPred(p,img){",
        "function drawPred(p,img){if(!S.prediction||p.prediction_id!==S.prediction.prediction_id)return;"
        "episodeROI=null;dragStart=null;episodeImage=img&&p.source_image?{prediction_id:p.prediction_id,image_sha256:p.source_image.file_sha256}:null;",
    )
    page = page.replace("clearRect(0,0,cv.width,cv.height);shownPred=null;",
                        "clearRect(0,0,cv.width,cv.height);episodeImage=episodeROI=dragStart=null;shownPred=null;")
    page = page.replace("strip();panelTest();", "strip();").replace(
        "if(Date.now()-lastReport", "episodePanel();if(Date.now()-lastReport"
    )
    page = page.replace(
        "No model output yet. Use the Test box above.",
        "Waiting for the first authorized episode observation.",
    )
    return page.replace("poll();pollLidar();", SCRIPT + "\npoll();pollLidar();")
