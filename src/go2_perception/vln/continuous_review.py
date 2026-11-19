"""Read-only continuous traces in the project's loopback VLN viewer."""

from base64 import b64decode
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from threading import Lock

from .continuous_recording import missing_continuation
from .interactive_page import REVIEW_STYLE
from .shadow_display import path_svg


PAGE = (
    """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Continuous navigation review</title>
<style>"""
    + REVIEW_STYLE
    + r"""
main{max-width:1200px}.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap}input[type=range]{flex:1;min-width:140px}
img{max-height:230px;object-fit:contain}h1{font-size:20px}.grid{grid-template-columns:repeat(auto-fit,minmax(min(100%,455px),1fr))}
#map{width:min(100%,392px);height:auto;margin:auto}#native svg{width:100%;height:245px}label{display:inline-flex;align-items:center;gap:6px}
</style></head><body><main><h1>Continuous navigation review</h1>
<section class="controls"><label>Run <select id="case"></select></label><label>Pass <select id="pass"></select></label>
<label><input type="checkbox" id="follow">Follow latest</label><button id="prev">Previous event</button><button id="next">Next event</button>
<label>Tick <input type="range" min="0" value="0" id="step"></label><output id="counter"></output></section>
<p id="title"></p><div class="grid"><section><h2>Camera observation</h2><img class="cam" id="current" alt="Latest saved camera observation"><pre id="camstamp" class="meta"></pre></section>
<section><h2>Incoming model proposal</h2><img class="cam" id="source" alt="Source image for latest delivered proposal"><pre class="meta" id="proposal"></pre></section>
<section><h2>Native waypoints at capture</h2><div id="native"></div></section>
<section><h2>Active segment and odometry</h2><canvas class="chart" id="map" width="480" height="300"></canvas>
<p class="note">Gray shows the measured path. Orange shows the selected primitive. Blue shows the current heading. Axes are odometry metres.</p></section></div>
<section><h2>Controller</h2><pre class="meta" id="diag"></pre></section></main><script>
const $=id=>document.getElementById(id), fmt=(x,n=3)=>x==null?'unknown':Number(x).toFixed(n);
const entry=new URLSearchParams(location.search);let cases, events=[], request=0;
function options(el,items){el.replaceChildren(...items.map(([v,t])=>{const o=document.createElement('option');o.value=v;o.textContent=t;return o}))}
function pic(id,img){const el=$(id);el.hidden=!img;if(img)el.src='data:image/png;base64,'+img;else el.removeAttribute('src')}
function map(d){const c=$('map'),g=c.getContext('2d');g.clearRect(0,0,c.width,c.height);
 const o=d.origin;if(!o)return;const scale=125,X=x=>240+(x-o.x_m)*scale,Y=y=>150-(y-o.y_m)*scale;
 function line(points,col,w){if(!points.length)return;g.strokeStyle=col;g.lineWidth=w;g.beginPath();points.forEach(([x,y],i)=>i?g.lineTo(X(x),Y(y)):g.moveTo(X(x),Y(y)));g.stroke()}
 function arrow(p,angle,l,col){line([[p.x_m,p.y_m],[p.x_m+l*Math.cos(angle),p.y_m+l*Math.sin(angle)]],col,3)}
 g.fillStyle='#666';g.fillText('+x →     +y ↑',10,17);line(d.path,'#888',2);
 const a=d.packet.action,b=d.packet.source_pose;
 if(a&&b){if(a.type==='forward')arrow(b,b.yaw_rad,a.magnitude,'#cf720a');
 else{const sign=a.direction==='left'?1:-1;g.beginPath();g.strokeStyle='#cf720a';g.lineWidth=4;g.arc(X(b.x_m),Y(b.y_m),26,-b.yaw_rad,-(b.yaw_rad+sign*a.magnitude),sign>0);g.stroke();arrow(b,b.yaw_rad+sign*a.magnitude,.34,'#cf720a')}}
 if(d.pose){g.fillStyle='#2368a2';g.beginPath();g.arc(X(d.pose.x_m),Y(d.pose.y_m),4,0,7);g.fill();arrow(d.pose,d.pose.yaw_rad,.30,'#2368a2')}}
async function draw(){const seq=++request;const q=new URLSearchParams({case:$('case').value,pass:$('pass').value,tick:$('follow').checked?'latest':$('step').value});
 const res=await fetch('/api/continuous?'+q);if(!res.ok)throw Error('Unable to load tick'); const d=await res.json();if(seq!==request)return;
 if(d.waiting){$('diag').textContent='Waiting for the first recorded control tick.';return}
 const p=d.packet,n=d.incoming;events=d.events;$('counter').textContent=(d.index+1)+' / '+d.count;$('step').max=d.count-1;$('step').value=d.index;
 $('title').textContent=d.case+' · '+d.pass+' · '+d.instruction;
 pic('current',d.camera);pic('source',d.model_image);$('camstamp').textContent=d.camera_id||'No saved view at this time';
 $('proposal').textContent=n? [n.id,'Source '+n.frame,(n.fixture_delay==null?'Worker '+fmt(n.latency):'Fixture delay '+fmt(n.fixture_delay))+' s · capture to result '+fmt(n.age)+' s',n.raw].join('\n'):'Awaiting first result';
 $('native').innerHTML=d.native_svg;map(d);
 $('diag').textContent=[p.mode+' · '+(p.handover||'handover unrecorded')+' · '+p.reason+(d.stop_detail?' ('+d.stop_detail+')':''),'Active '+(p.prediction_id||'none')+' · outstanding '+(d.pending||'none'),
 'Velocity '+fmt(p.velocity[0])+' m/s · yaw '+fmt(p.velocity[1])+' rad/s',
 'Remaining '+fmt(p.remaining_path)+' '+(p.action?.unit||'')+' · to brake '+fmt(p.remaining_to_brake)+' · plan horizon '+fmt(p.plan_horizon_s)+' s',
 'Pose x '+fmt(d.pose?.x_m)+' m, y '+fmt(d.pose?.y_m)+' m, heading '+fmt(d.pose?.yaw_rad)+' rad',
 'Tick '+fmt(p.t)+' s · plans '+p.plans+' · moving replacements '+p.rollovers].join('\n')}
async function selectCase(initial=false){const c=cases[Number($('case').value)];options($('pass'),c.passes.map(p=>[p,p]));if(initial&&c.passes.includes(entry.get('pass')))$('pass').value=entry.get('pass');
 $('step').max=4095;$('step').value=initial?Math.max(0,Number(entry.get('tick')||0)):0;await draw()}
function moveEvent(sign){const i=Number($('step').value);$('follow').checked=false;const k=(sign>0?events.find(x=>x>i):[...events].reverse().find(x=>x<i));if(k!=null){$('step').value=k;draw()}}
$('prev').onclick=()=>moveEvent(-1);$('next').onclick=()=>moveEvent(1);$('case').onchange=()=>selectCase();$('pass').onchange=()=>{$('step').value=0;draw()};$('step').oninput=()=>{$('follow').checked=false;draw()};$('follow').onchange=draw;
(async()=>{const state=await (await fetch('/api/state')).json();cases=state.cases;$('follow').checked=state.following&&!entry.has('tick');options($('case'),cases.map((c,i)=>[i,c.label]));const chosen=Number(entry.get('case')||0);if(Number.isInteger(chosen)&&cases[chosen])$('case').value=chosen;await selectCase(true);async function refresh(){try{if($('follow').checked)await draw()}finally{setTimeout(refresh,250)}}setTimeout(refresh,250)})().catch(e=>{$('diag').textContent=e.message});
</script></body></html>"""
)


class TraceIndex:
    """Keep offsets and compact poses. Load images and full packets from disk on demand."""

    max_rows = 4096

    def __init__(self, path):
        self.path = path
        self.steps = defaultdict(list)
        self.events = defaultdict(list)
        self.points = defaultdict(list)
        self.frames = {}
        self.camera = self.incoming = self.source = None
        self.position = self.rows = 0

    def refresh(self):
        if not self.path.is_file():
            return
        with self.path.open("rb") as stream:
            stream.seek(self.position)
            while True:
                offset = stream.tell()
                line = stream.readline(2 * 1024 * 1024 + 1)
                if not line:
                    break
                if self.rows >= self.max_rows or len(line) > 2 * 1024 * 1024:
                    raise ValueError("trace exceeds the episode review bound")
                if not line.endswith(b"\n"):
                    break
                self.position = stream.tell()
                self.rows += 1
                row = json.loads(line)
                if row["kind"] == "archived_proposal":
                    self.camera = self.incoming = self.source = offset
                if row["kind"] == "capture":
                    self.frames[row["observation_id"]] = self.camera = offset
                if row["kind"] != "tick":
                    continue
                key = row.get("counterfactual_pass", "simulation")
                if row.get("delivery"):
                    self.incoming = offset
                    self.source = self.frames.get(row["delivery"]["proposal"]["identity"]["observation_id"])
                p = row["packet"]
                pose = row.get("pose") or (row.get("guard") or {}).get("snapshot", {}).get("pose")
                steps = self.steps[key]
                self.points[key].append(None if pose is None else [pose["x_m"], pose["y_m"]])
                marker = (p.get("handover"), p["reason"], p["prediction_id"])
                if not steps or marker != steps[-1][-1] or row.get("delivery"):
                    self.events[key].append(len(steps))
                steps.append((offset, self.camera, self.incoming, self.source, marker))

    def sample(self, group, index, label):
        self.refresh()
        if index == "latest":
            if not self.steps.get(group):
                return {"waiting": True}
            index = len(self.steps[group]) - 1
        if group not in self.steps or not 0 <= index < len(self.steps[group]):
            raise ValueError("unknown trace position")
        off, cam, inc, src, _ = self.steps[group][index]
        with self.path.open("rb") as f:

            def get(offset):
                if offset is None:
                    return {}
                f.seek(offset)
                return json.loads(f.readline(2 * 1024 * 1024 + 1))

            tick, current, latest, source = (get(k) for k in (off, cam, inc, src))
            prior = (
                get(self.steps[group][index - 1][0]).get("packet")
                if index and tick["packet"].get("handover") == "stop"
                else None
            )

        def image(row, expected=None):
            v = row.get("view", {})
            if not v.get("png_b64"):
                return None
            if sha256(b64decode(v["png_b64"], validate=True)).hexdigest() != v["png_sha256"]:
                raise ValueError("review image hash mismatch")
            if expected and expected not in (v.get("pixel_sha256"), v["png_sha256"]):
                raise ValueError("proposal and frame hash disagree")
            return v["png_b64"]

        delivery = latest.get("delivery") or {}
        p = delivery.get("proposal", {})
        ident = p.get("identity", {})
        pose = tick.get("pose") or (tick.get("guard") or {}).get("snapshot", {}).get("pose")
        points = [pt for pt in self.points[group][: index + 1] if pt is not None]
        incoming = (
            None
            if not p
            else dict(
                id=ident["prediction_id"],
                frame=ident["observation_id"],
                raw=p["native_output"],
                latency=delivery.get("inference_s"),
                fixture_delay=delivery.get("fixture_delay_s"),
                age=delivery["available_s"] - ident["observed_monotonic_s"],
            )
        )
        return dict(
            stop_detail=missing_continuation(tick["packet"]["reason"], prior) if prior else None,
            index=index,
            count=len(self.steps[group]),
            events=self.events[group],
            **{"case": label, "pass": group},
            instruction=ident.get("instruction", ""),
            pending=tick.get("pending_observation"),
            camera_id=current.get("observation_id") or (ident.get("observation_id") if current else None),
            camera=image(current),
            model_image=image(source, ident.get("source_image_sha256")),
            native_svg=path_svg(p.get("waypoints")),
            incoming=incoming,
            packet=tick["packet"],
            pose=pose,
            path=points,
            origin={"x_m": points[0][0], "y_m": points[0][1]} if points else None,
        )


class ContinuousReview:
    page = PAGE

    def __init__(self, root, summary):
        self.following = summary.get("schema") == "go2-continuous-review-v1"
        self.files = []
        self.cases = []
        self.index = None
        self.active = None
        self.lock = Lock()
        for r in summary["results"]:
            path = (root / r["trace"]).resolve()
            if path.parent != root.resolve():
                raise ValueError("trace must be in run directory")
            self.files.append(path)
            groups = list(r.get("passes", {"simulation": None}))
            label = r.get("scenario") or Path(r["recording"]).parent.name
            self.cases.append(dict(label=label, passes=groups))
        if not self.cases:
            raise ValueError("no recorded cases")

    def state(self, now):
        return dict(cases=self.cases, following=self.following)

    def continuous_sample(self, q):
        case = int(q.get("case", 0))
        if not 0 <= case < len(self.files):
            raise ValueError("unknown case")
        with self.lock:
            if self.active != case:
                self.index = TraceIndex(self.files[case])
                self.active = case
            spec = self.cases[case]
            return self.index.sample(
                q.get("pass", spec["passes"][0]),
                "latest" if q.get("tick") == "latest" else int(q.get("tick", 0)),
                spec["label"],
            )


def review_loop(root, summary, port):
    from .interactive_server import DisplayServer

    server = DisplayServer(ContinuousReview(root, summary), port)
    server.start()
    print(f"Read-only continuous navigation review: {server.url}", flush=True)
    try:
        while input("Press Enter or type quit to close: ").strip() not in ("", "quit"):
            pass
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        server.close()
    return 0
