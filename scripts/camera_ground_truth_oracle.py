#!/usr/bin/env python3
"""Camera ground-truth oracle for RuView — automated labelled capture + live scoring.

Uses the USB webcam + MediaPipe pose detection as the authoritative occupancy
oracle. Runs unattended for hours. Two jobs, no human coordination:

  1. AUTO-LABEL + CAPTURE: watches person-present vs person-absent (debounced,
     with a guard window around transitions so the ambiguous edges are never
     recorded). While a state is STABLE it records correctly-labelled CSI via
     RuView's /api/v1/recording endpoint — up to a bounded budget per class so
     the disk doesn't fill. Files: auto_empty_<ts>.jsonl / auto_occupied_<ts>.jsonl.

  2. LIVE SCORING: polls RuView /api/v1/sensing/latest and compares its
     `presence` to the camera truth (only while stable, outside guard windows),
     maintaining a live confusion matrix. This is the automated TEST — it tells
     us RuView's real false-positive rate (says present while camera says empty)
     and false-negative rate, continuously.

Camera labels "person in view". Aim it to cover the work area; a person in the
room but fully out of frame will read absent (logged so it's visible).

Usage:
    python scripts/camera_ground_truth_oracle.py            # run until Ctrl-C
    python scripts/camera_ground_truth_oracle.py --minutes 60
    python scripts/camera_ground_truth_oracle.py --no-record # score only, no CSI capture
"""
from __future__ import annotations
import argparse, json, os, shutil, signal, subprocess, sys, time, urllib.request, threading
from collections import deque, Counter
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import cv2, numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
import mediapipe as mp

BASE = "http://127.0.0.1:3000"
MODEL = "data/.cache/pose_landmarker_lite.task"
OUT   = Path(os.environ.get("ORACLE_OUT",
        "/home/davestanyer/.local/share/ruview/oracle"))
OUT.mkdir(parents=True, exist_ok=True)

DETECT_FPS      = 3.0
VIS_MIN         = 0.30    # mean landmark visibility to count as a real detection
# Asymmetric hysteresis: a person doesn't blink out of existence. Go OCCUPIED
# fast (any detection within OCC_HOLD), but only declare EMPTY after a sustained
# gap (no detection for EMPTY_CONFIRM) so MediaPipe dropouts can't fake empty.
OCC_HOLD        = 4.0     # stay occupied if seen within this many secs
EMPTY_CONFIRM   = 12.0    # need this many secs with NO detection to call empty
STABLE_SECS     = 15.0    # must hold a state this long before it counts (CSI hysteresis settle)
GUARD_SECS      = 12.0    # after any transition, don't score or record
SCORE_POLL      = 4.0     # seconds between RuView presence polls
FACE_DIR        = Path(os.environ.get("ORACLE_FACE_DIR", "/home/davestanyer/RuView/tools/jarvis-face"))
FACE_ADDR       = os.environ.get("ORACLE_FACE_ADDR", "54:74:52:82:48:C4")
FACE_WAVE_SECS  = float(os.environ.get("ORACLE_FACE_WAVE_SECS", "10"))
# Disabled by default: the dedicated ruview_face_status.py watcher owns BLE writes.
# Oracle-side GIF/image hooks compete for the same POPLED device and can trigger
# BlueZ `Operation already in progress` / service-discovery failures.
FACE_ENABLED    = os.environ.get("ORACLE_FACE_ENABLED", "0") != "0"
# R3: refresh the server's empty baseline whenever the camera confirms empty,
# rate-limited to stay well inside the gate's 12-min staleness horizon.
BASELINE_REFRESH_SECS = 600.0
BASELINE_CAPTURE_SECS = 90.0   # R3 iteration 2026-07-08: 30->90s (truer spread)
REC_BUDGET_SECS = 7200.0  # default max CSI seconds to record PER class (CLI-overridable
                          # via --empty-budget / --occ-budget for long unattended runs)
REC_SEG_SECS    = 300.0   # rotate recording file every N secs

def post(path, body=None):
    data=json.dumps(body).encode() if body is not None else b""
    req=urllib.request.Request(BASE+path, data=data, headers={"Content-Type":"application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=6) as r: return json.load(r)

def get_presence():
    with urllib.request.urlopen(BASE+"/api/v1/sensing/latest", timeout=5) as r:
        d=json.load(r)
    return (d.get("classification") or {}).get("presence")

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

_face_lock = threading.Lock()

def _disconnect_face_panel():
    if not FACE_ADDR:
        return
    try:
        subprocess.run(["bluetoothctl", "disconnect", FACE_ADDR],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=6)
        time.sleep(1.0)
    except Exception:
        pass

def _face_upload(kind, asset):
    cmd = ["uv", "run", "face.py", kind, asset]
    return subprocess.run(cmd, cwd=FACE_DIR, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, timeout=45)

def _c3po_presence_greeting():
    if not FACE_ENABLED:
        return
    if not FACE_DIR.joinpath("face.py").exists():
        log(f"face hook skipped: {FACE_DIR}/face.py not found")
        return
    if not _face_lock.acquire(blocking=False):
        log("face hook skipped: C-3PO sequence already running")
        return
    try:
        log("face hook: C-3PO wave")
        _disconnect_face_panel()
        r = _face_upload("gif", "art/c3po_wave.gif")
        if r.returncode != 0:
            log(f"face hook wave failed: {r.stdout.strip()[-500:]}")
            return
        time.sleep(FACE_WAVE_SECS)
        log("face hook: C-3PO main face")
        _disconnect_face_panel()
        r = _face_upload("image", "art/c3po_face.png")
        if r.returncode != 0:
            log(f"face hook main face failed: {r.stdout.strip()[-500:]}")
    except Exception as e:
        log(f"face hook error: {e}")
    finally:
        _face_lock.release()

def trigger_c3po_presence_greeting():
    threading.Thread(target=_c3po_presence_greeting, daemon=True).start()

# ---- Paired live view: camera (ground truth) + RuView verdict, over MJPEG ----
# Reuses the frames the oracle already grabs, so no camera contention. Serves a
# page at http://127.0.0.1:<port>/ showing the webcam with the camera-detected
# state and RuView's live presence/gate verdict burned in — the honest "watch it
# work" view (the Observatory silently falls back to demo data on WS drop).
_frame_lock = threading.Lock()
_latest_jpeg = [None]
_overlay = {"cam": "?", "ruview": "?", "gate": "?", "voting": 0, "occ": 0,
            "base_age": None, "node_z": {}, "nodes": {}}
_pose3d = {"kp": None, "present": False}   # live metric COCO-17 skeleton (meters, hip-centred)
_wifi = {
    "present": False,
    "pos": None,
    "motion": "?",
    "persons": 0,
    "verdict": "?",
    "nodes_voting": 0,
    "nodes_occupied": 0,
    "baseline_stale": None,
    "desk_state": "?",
    "desk_confidence": 0.0,
    "desk_seated": False,
    "model_seated": 0.0,
    "model_standing_near": 0.0,
    "model_away": 0.0,
}  # WiFi-inferred state
STREAM_K = 8.0  # per-node deviation threshold (mirrors presence_baseline.rs K)

# Room geometry (Zugg's office map, 2026-07-08). Metres. Origin NW/top-left;
# X east, Y south (depth), Z up. Desk where Dave sits.
ROOM = {"w": 4.14, "d": 5.0, "h": 2.6, "desk_x": 1.32, "desk_y": 1.25, "seat_h": 0.85}
COCO_BONES = [[5,7],[7,9],[6,8],[8,10],[5,6],[11,13],[13,15],[12,14],[14,16],
              [11,12],[5,11],[6,12],[0,1],[0,2],[1,3],[2,4]]

# MediaPipe-33 -> COCO-17 keypoint mapping (matches ADR-079 collect-ground-truth.py
# so the existing align/train pipeline can consume this corpus directly).
MP_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
POSE_DIR = Path("/home/davestanyer/.local/share/ruview/data/pose_corpus")

_PAGE = b"""<!doctype html><html><head><title>RuView Paired View</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{margin:0;background:#0d0d0f;color:#e8e8ea;font-family:system-ui;text-align:center}
h2{margin:10px;font-weight:600}img{max-width:100%;height:auto;border-radius:6px}
.legend{color:#9aa;font-size:14px;margin:8px}</style></head>
<body><h2>RuView &mdash; Camera (ground truth) vs Detection</h2>
<img src="/stream" alt="live"><div class="legend">Camera state and RuView presence are burned into the frame, updating live.<br>
<a href="/3d" style="color:#4ea1ff">&#9654; Live 3D skeleton</a> &nbsp;|&nbsp;
<a href="/compare" style="color:#f2a154">&#9654; Camera vs WiFi side-by-side</a></div>
</body></html>"""


_3D_PAGE = b"""<!doctype html><html><head><title>RuView 3D</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{margin:0;background:#0b0b0e;color:#ddd;font-family:system-ui;overflow:hidden}
#hud{position:fixed;top:8px;left:10px;font-size:14px}#hud b{color:#4ea1ff}
canvas{display:block}</style></head><body>
<div id="hud"><b>RuView</b> &mdash; live 3D skeleton (from camera, ground truth) &middot; <span id="st">...</span>
<div style="font-size:12px;color:#889">drag to rotate &middot; scroll to zoom</div></div>
<canvas id="c"></canvas><script>
const cv=document.getElementById('c'),cx=cv.getContext('2d');
let W,H;function rs(){W=cv.width=innerWidth;H=cv.height=innerHeight}addEventListener('resize',rs);rs();
let yaw=-0.6,pitch=0.5,dist=7,room=null,bones=[],kp=null,present=false;
let drag=false,px,py;
cv.onmousedown=e=>{drag=true;px=e.clientX;py=e.clientY};
onmouseup=()=>drag=false;
onmousemove=e=>{if(drag){yaw+=(e.clientX-px)*0.008;pitch=Math.max(-1.4,Math.min(1.4,pitch+(e.clientY-py)*0.008));px=e.clientX;py=e.clientY}};
cv.onwheel=e=>{dist=Math.max(3,Math.min(14,dist+e.deltaY*0.005));e.preventDefault()};
// world->screen (orbit camera around room centre)
function proj(p){const cxr=room?room.w/2:2,cyr=room?room.d/2:2.5,czr=1.0;
 let x=p[0]-cxr,y=p[1]-cyr,z=p[2]-czr;
 const cy=Math.cos(yaw),sy=Math.sin(yaw);let x1=x*cy-y*sy,y1=x*sy+y*cy;
 const cp=Math.cos(pitch),sp=Math.sin(pitch);let z1=z*cp-y1*sp,y2=z*sp+y1*cp;
 const zc=y2+dist;if(zc<0.2)return null;const f=H*0.9/zc;
 return [W/2+x1*f, H/2-z1*f, zc]}
function line(a,b,col,wd){const pa=proj(a),pb=proj(b);if(!pa||!pb)return;
 cx.strokeStyle=col;cx.lineWidth=wd||1;cx.beginPath();cx.moveTo(pa[0],pa[1]);cx.lineTo(pb[0],pb[1]);cx.stroke()}
function draw(){cx.clearRect(0,0,W,H);
 if(room){const w=room.w,d=room.d,h=room.h;const c='#26303a';
  const corners=[[0,0,0],[w,0,0],[w,d,0],[0,d,0]];
  for(let i=0;i<4;i++){line(corners[i],corners[(i+1)%4],c,1);
   line(corners[i],[corners[i][0],corners[i][1],h],c,1);
   line([corners[i][0],corners[i][1],h],[corners[(i+1)%4][0],corners[(i+1)%4][1],h],c,1);}
  // desk marker
  const dx=room.desk_x,dy=room.desk_y;line([dx-.4,dy-.3,0],[dx+.4,dy-.3,0],'#3a6',2);
  line([dx-.4,dy-.3,0],[dx-.4,dy+.3,0],'#3a6',2);line([dx+.4,dy-.3,0],[dx+.4,dy+.3,0],'#3a6',2);line([dx-.4,dy+.3,0],[dx+.4,dy+.3,0],'#3a6',2);}
 if(present&&kp){for(const [a,b] of bones){if(kp[a]&&kp[b])line(kp[a],kp[b],'#4ea1ff',3)}
  for(const p of kp){if(!p)continue;const s=proj(p);if(s){cx.fillStyle='#7fd';cx.beginPath();cx.arc(s[0],s[1],3,0,7);cx.fill()}}}
 requestAnimationFrame(draw)}
async function poll(){try{const r=await fetch('/pose.json');const j=await r.json();
 room=j.room;bones=j.bones;present=j.present;
 document.getElementById('st').textContent=present?'PERSON (tracking)':'empty (no skeleton)';
 if(j.kp&&present){const w=j.room,dx=w.desk_x,dy=w.desk_y,sh=w.seat_h;
  // MediaPipe world landmarks: metres, hip-centred, y-down, z toward camera.
  // Place hip at the desk; map (mpx->X, -mpy->Z up, mpz->Y depth).
  kp=j.kp.map(p=>p?[dx+p[0], dy+p[2], sh-p[1]]:null)}else kp=null;
 }catch(e){}}
setInterval(poll,150);poll();draw();
</script></body></html>"""


_COMPARE_PAGE = b"""<!doctype html><html><head><title>RuView: Camera vs WiFi</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{margin:0;background:#0b0b0e;color:#ddd;font-family:system-ui;overflow:hidden}
#wrap{display:flex;height:100vh}.pane{flex:1;position:relative;border-right:1px solid #222}
.lbl{position:absolute;top:8px;left:12px;font-size:15px;font-weight:600}
.sub{position:absolute;top:30px;left:12px;font-size:12px;color:#8a8}
#c3po{position:absolute;right:14px;top:14px;width:255px;padding:12px 14px;border:1px solid #3a3320;border-radius:14px;background:rgba(15,13,10,.86);box-shadow:0 10px 35px rgba(0,0,0,.35);text-align:left}
#c3po .head{display:flex;gap:10px;align-items:center;margin-bottom:8px}.face{width:42px;height:42px;border-radius:50%;display:grid;place-items:center;background:linear-gradient(145deg,#f8d66a,#8b681d);color:#1a1307;font-size:24px;font-weight:800}.name{color:#f8d66a;font-weight:800;letter-spacing:.04em}.line{display:flex;justify-content:space-between;gap:10px;margin:5px 0;font-size:13px}.k{color:#9b9279}.v{color:#eee;font-weight:650}.ok{color:#7ff0ac}.warn{color:#f8d66a}.bad{color:#ff7474}.big{font-size:18px;font-weight:850;color:#fff;margin:4px 0 8px}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:7px;background:#777;box-shadow:0 0 12px #777}.dot.ok{background:#37f27c;box-shadow:0 0 14px #37f27c}.dot.bad{background:#ff4747;box-shadow:0 0 14px #ff4747}.dot.warn{background:#f8d66a;box-shadow:0 0 14px #f8d66a}
canvas{display:block}</style></head><body>
<div id="wrap">
 <div class="pane"><div class="lbl" style="color:#4ea1ff">CAMERA (ground truth)</div>
   <div class="sub" id="cs">17-joint skeleton</div><canvas id="cA"></canvas></div>
 <div class="pane"><div class="lbl" style="color:#f2a154">WiFi CSI (RuView)</div>
   <div class="sub" id="ws">presence + position &middot; pose model not trained yet</div><canvas id="cB"></canvas>
   <div id="c3po"><div class="head"><div class="face">C</div><div><div class="name">C-3PO STATUS</div><div style="font-size:12px;color:#9b9279">protocol droid for the RF field</div></div></div>
     <div class="big" id="robotStatus">Awaiting telemetry...</div>
     <div class="line"><span class="k">People</span><span class="v" id="robotPeople">--</span></div>
     <div class="line"><span class="k">Camera agree</span><span class="v"><span id="robotAgreeDot" class="dot warn"></span><span id="robotAgree">--</span></span></div>
     <div class="line"><span class="k">Desk</span><span class="v" id="robotDesk">--</span></div>
     <div class="line"><span class="k">Confidence</span><span class="v" id="robotConf">--</span></div>
     <div class="line"><span class="k">Model votes</span><span class="v" id="robotVotes">--</span></div>
   </div></div>
</div><script>
let yaw=-0.6,pitch=0.5,dist=7,data=null,drag=false,px,py;
addEventListener('mousedown',e=>{drag=true;px=e.clientX;py=e.clientY});
addEventListener('mouseup',()=>drag=false);
addEventListener('mousemove',e=>{if(drag){yaw+=(e.clientX-px)*0.008;pitch=Math.max(-1.4,Math.min(1.4,pitch+(e.clientY-py)*0.008));px=e.clientX;py=e.clientY}});
addEventListener('wheel',e=>{dist=Math.max(3,Math.min(14,dist+e.deltaY*0.005))});
function mk(id){const c=document.getElementById(id);const x=c.getContext('2d');
 function rs(){c.width=c.parentElement.clientWidth;c.height=c.parentElement.clientHeight}rs();addEventListener('resize',rs);return {c,x}}
const A=mk('cA'),B=mk('cB');
function proj(c,p,room){const cxr=room.w/2,cyr=room.d/2,czr=1.0;
 let x=p[0]-cxr,y=p[1]-cyr,z=p[2]-czr;
 const cy=Math.cos(yaw),sy=Math.sin(yaw);let x1=x*cy-y*sy,y1=x*sy+y*cy;
 const cp=Math.cos(pitch),sp=Math.sin(pitch);let z1=z*cp-y1*sp,y2=z*sp+y1*cp;
 const zc=y2+dist;if(zc<0.2)return null;const f=c.height*0.85/zc;
 return [c.width/2+x1*f,c.height/2-z1*f]}
function ln(ctx,c,a,b,room,col,w){const pa=proj(c,a,room),pb=proj(c,b,room);if(!pa||!pb)return;
 ctx.strokeStyle=col;ctx.lineWidth=w||1;ctx.beginPath();ctx.moveTo(pa[0],pa[1]);ctx.lineTo(pb[0],pb[1]);ctx.stroke()}
function room3(ctx,c,room){const w=room.w,d=room.d,h=room.h,col='#26303a';
 const cr=[[0,0,0],[w,0,0],[w,d,0],[0,d,0]];
 for(let i=0;i<4;i++){ln(ctx,c,cr[i],cr[(i+1)%4],room,col,1);ln(ctx,c,cr[i],[cr[i][0],cr[i][1],h],room,col,1);
  ln(ctx,c,[cr[i][0],cr[i][1],h],[cr[(i+1)%4][0],cr[(i+1)%4][1],h],room,col,1);}
 const dx=room.desk_x,dy=room.desk_y;ln(ctx,c,[dx-.4,dy-.3,0],[dx+.4,dy-.3,0],room,'#3a6',2);
 ln(ctx,c,[dx-.4,dy+.3,0],[dx+.4,dy+.3,0],room,'#3a6',2);ln(ctx,c,[dx-.4,dy-.3,0],[dx-.4,dy+.3,0],room,'#3a6',2);ln(ctx,c,[dx+.4,dy-.3,0],[dx+.4,dy+.3,0],room,'#3a6',2);}
function drawCam(){const {c,x}=A;x.clearRect(0,0,c.width,c.height);if(!data)return;const room=data.room;room3(x,c,room);
 if(data.present&&data.kp){const dx=room.desk_x,dy=room.desk_y,sh=room.seat_h;
  const kp=data.kp.map(p=>p?[dx+p[0],dy+p[2],sh-p[1]]:null);
  for(const [a,b] of data.bones){if(kp[a]&&kp[b])ln(x,c,kp[a],kp[b],room,'#4ea1ff',3)}
  for(const p of kp){if(p){const s=proj(c,p,room);if(s){x.fillStyle='#7fd';x.beginPath();x.arc(s[0],s[1],3,0,7);x.fill()}}}}
 document.getElementById('cs').textContent=data.present?'17-joint skeleton (tracking)':'no person';}
function drawWifi(){const {c,x}=B;x.clearRect(0,0,c.width,c.height);if(!data)return;const room=data.room;room3(x,c,room);
 const wf=data.wifi||{};
 if(wf.present){// WiFi knows WHERE + WHETHER, not pose yet: draw a column at its position
  let px=room.desk_x,py=room.desk_y; if(wf.pos){px=room.w/2+wf.pos[0];py=room.d/2+wf.pos[2];}
  const col=wf.verdict==='occupied'?'#f2a154':(wf.verdict==='empty'?'#e55':'#aa8');
  ln(x,c,[px,py,0],[px,py,1.7],room,col,10);
  const s=proj(c,[px,py,1.8],room);if(s){x.fillStyle=col;x.font='13px system-ui';x.fillText('~person',s[0]+8,s[1])}}
 const people=Number.isFinite(wf.persons)?wf.persons:0;
 const desk=wf.desk_state||'?';
 const conf=Math.round((wf.desk_confidence||0)*100);
 const seated=Math.round((wf.model_seated||0)*100), standing=Math.round((wf.model_standing_near||0)*100), away=Math.round((wf.model_away||0)*100);
 const camOcc=!!data.present;
 const wifiOcc=(people>0&&wf.motion!=='absent')||(wf.present===true&&wf.motion!=='absent');
 const agree=camOcc===wifiOcc;
 let status='Room appears empty', cls='bad';
 if(people>0&&desk==='desk_seated'){status='Human seated at desk';cls='ok'}
 else if(people>0&&desk==='near_desk_not_seated'){status='Human near desk, not seated';cls='warn'}
 else if(people>0){status='Human present';cls='warn'}
 document.getElementById('ws').textContent=(wf.present?('PRESENT ('+(wf.motion||'?')+', gate '+(wf.verdict||'?')+')'):'EMPTY')+' - '+people+' person(s) - desk '+desk;
 document.getElementById('robotStatus').textContent=status;
 document.getElementById('robotStatus').className='big '+cls;
 document.getElementById('robotPeople').textContent=String(people);
 document.getElementById('robotAgreeDot').className='dot '+(agree?'ok':'bad');
 document.getElementById('robotAgree').textContent=agree?'yes':'no';
 document.getElementById('robotDesk').textContent=desk+(wf.desk_seated?' yes':'');
 document.getElementById('robotConf').textContent=conf+'%';
 document.getElementById('robotVotes').textContent='S '+seated+' / St '+standing+' / A '+away;
}
async function poll(){try{data=await (await fetch('/pose.json')).json()}catch(e){}}
setInterval(poll,150);poll();
function loop(){drawCam();drawWifi();requestAnimationFrame(loop)}loop();
</script></body></html>"""


_C3PO_PAGE = b"""<!doctype html><html><head><title>RuView C-3PO Status</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{margin:0;background:#090806;color:#eee;font-family:system-ui;display:grid;place-items:center;min-height:100vh}.card{width:min(92vw,520px);padding:24px;border:1px solid #3a3320;border-radius:22px;background:linear-gradient(180deg,#17130b,#0f0d0a);box-shadow:0 20px 70px rgba(0,0,0,.5)}.head{display:flex;gap:16px;align-items:center}.face{width:74px;height:74px;border-radius:50%;display:grid;place-items:center;background:linear-gradient(145deg,#f8d66a,#8b681d);color:#1a1307;font-size:42px;font-weight:900}.name{color:#f8d66a;font-weight:900;letter-spacing:.05em}.sub{color:#9b9279}.status{font-size:30px;line-height:1.1;font-weight:900;margin:22px 0}.ok{color:#7ff0ac}.warn{color:#f8d66a}.bad{color:#ff7474}.row{display:flex;justify-content:space-between;border-top:1px solid #2a2418;padding:12px 0}.k{color:#9b9279}.v{font-weight:800}.dot{display:inline-block;width:12px;height:12px;border-radius:50%;margin-right:8px;background:#777;box-shadow:0 0 14px #777}.dot.ok{background:#37f27c;box-shadow:0 0 18px #37f27c}.dot.bad{background:#ff4747;box-shadow:0 0 18px #ff4747}.dot.warn{background:#f8d66a;box-shadow:0 0 18px #f8d66a}.links{margin-top:18px;font-size:13px}.links a{color:#f8d66a;margin-right:14px}</style></head><body>
<div class="card"><div class="head"><div class="face">C</div><div><div class="name">C-3PO STATUS</div><div class="sub">RuView RF field interpreter</div></div></div>
<div id="status" class="status warn">Awaiting telemetry...</div>
<div class="row"><span class="k">People detected</span><span id="people" class="v">--</span></div>
<div class="row"><span class="k">Camera agreement</span><span class="v"><span id="agreeDot" class="dot warn"></span><span id="agree">--</span></span></div>
<div class="row"><span class="k">Desk state</span><span id="desk" class="v">--</span></div>
<div class="row"><span class="k">Confidence</span><span id="conf" class="v">--</span></div>
<div class="row"><span class="k">Model votes</span><span id="votes" class="v">--</span></div>
<div class="row"><span class="k">Motion</span><span id="motion" class="v">--</span></div>
<div class="links"><a href="/">paired view</a><a href="/compare">camera vs wifi</a><a href="/pose.json">json</a></div></div>
<script>
async function tick(){const el=id=>document.getElementById(id);try{const j=await (await fetch('/pose.json')).json();const w=j.wifi||{};const p=Number.isFinite(w.persons)?w.persons:0;const d=w.desk_state||'?';const c=Math.round((w.desk_confidence||0)*100);const s=Math.round((w.model_seated||0)*100),st=Math.round((w.model_standing_near||0)*100),a=Math.round((w.model_away||0)*100);const camOcc=!!j.present;const wifiOcc=(p>0&&w.motion!=='absent')||(w.present===true&&w.motion!=='absent');const agree=camOcc===wifiOcc;let text='Room appears empty',cls='status bad';if(p>0&&d==='desk_seated'){text='Human seated at desk';cls='status ok'}else if(p>0&&d==='near_desk_not_seated'){text='Human near desk, not seated';cls='status warn'}else if(p>0){text='Human present';cls='status warn'}el('status').textContent=text;el('status').className=cls;el('people').textContent=p;el('agreeDot').className='dot '+(agree?'ok':'bad');el('agree').textContent=agree?(camOcc?'yes - both occupied':'yes - both empty'):(camOcc?'no - camera occupied / CSI empty':'no - camera empty / CSI present');el('desk').textContent=d+(w.desk_seated?' yes':'');el('conf').textContent=c+'%';el('votes').textContent='S '+s+' / St '+st+' / A '+a;el('motion').textContent=w.motion||'?'}catch(e){el('status').textContent='Telemetry unavailable';el('status').className='status bad'}}
setInterval(tick,500);tick();
</script></body></html>"""


class _StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(_PAGE)
            return
        if path == "/3d":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(_3D_PAGE)
            return
        if path == "/compare":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(_COMPARE_PAGE)
            return
        if path == "/c3po":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(_C3PO_PAGE)
            return
        if path == "/pose.json":
            with _frame_lock:
                body = json.dumps({"kp": _pose3d["kp"], "present": _pose3d["present"],
                                   "room": ROOM, "bones": COCO_BONES,
                                   "wifi": dict(_wifi)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with _frame_lock:
                        jpg = _latest_jpeg[0]
                    if jpg:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(jpg)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.1)
            except (BrokenPipeError, ConnectionResetError):
                return
        self.send_response(404)
        self.end_headers()


def _start_stream_server(port):
    srv = ThreadingHTTPServer(("127.0.0.1", port), _StreamHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _publish_frame(cv2, np, frame, landmarks):
    """Compose camera (with pose) + a WiFi-ESP32 telemetry panel, and publish.

    Left: webcam with pose dots and the CAMERA vs RuView verdicts + MATCH.
    Right: per-node CSI deviation (z vs empty baseline), vote state, RSSI —
    the WiFi side and the inferred match, side by side with ground truth.
    """
    _cv2 = cv2
    GREEN, RED, GREY, AMBER = (0, 220, 0), (60, 60, 230), (150, 150, 150), (0, 170, 235)
    f = frame.copy()
    h, w = f.shape[:2]
    if landmarks:
        for l in landmarks:
            if getattr(l, "visibility", 1.0) > 0.4:
                _cv2.circle(f, (int(l.x * w), int(l.y * h)), 3, GREEN, -1)
    cam, rv, gate = _overlay["cam"], _overlay["ruview"], _overlay["gate"]
    cam_c = GREEN if cam == "person" else (RED if cam == "empty" else GREY)
    rv_c = GREEN if rv == "PRESENT" else (RED if rv == "EMPTY" else GREY)
    # header band on the camera
    _cv2.rectangle(f, (0, 0), (w, 40), (20, 20, 24), -1)
    _cv2.putText(f, f"CAMERA: {cam}", (10, 28), _cv2.FONT_HERSHEY_SIMPLEX, 0.7, cam_c, 2)
    # match badge (bottom of camera)
    cam_present = cam == "person"; rv_present = rv == "PRESENT"
    match = (cam_present == rv_present) if rv in ("PRESENT", "EMPTY") else None
    if match is not None:
        _cv2.rectangle(f, (0, h - 34), (w, h), (20, 20, 24), -1)
        _cv2.putText(f, "MATCH" if match else "MISMATCH", (10, h - 10),
                     _cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN if match else RED, 2)

    # ---- WiFi telemetry panel (right side) ----
    PW = 340
    panel = np.full((h, PW, 3), 18, np.uint8)
    y = 30
    _cv2.putText(panel, "WiFi CSI (ESP32)", (12, y), _cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 2); y += 30
    _cv2.putText(panel, f"RuView: {rv}", (12, y), _cv2.FONT_HERSHEY_SIMPLEX, 0.65, rv_c, 2); y += 24
    with _frame_lock:
        wf = dict(_wifi)
    desk = wf.get("desk_state", "?")
    persons = int(wf.get("persons") or 0)
    desk_conf = float(wf.get("desk_confidence") or 0.0)
    votes = (float(wf.get("model_seated") or 0.0),
             float(wf.get("model_standing_near") or 0.0),
             float(wf.get("model_away") or 0.0))
    status = "seated" if wf.get("desk_seated") else ("near desk" if desk == "near_desk_not_seated" else desk)
    _cv2.putText(panel, f"C-3PO: {persons} person(s), {status}",
                 (12, y), _cv2.FONT_HERSHEY_SIMPLEX, 0.52, (120, 240, 170) if persons else RED, 1); y += 21
    _cv2.putText(panel, f"desk {desk_conf*100:.0f}%  S/St/A {votes[0]*100:.0f}/{votes[1]*100:.0f}/{votes[2]*100:.0f}",
                 (12, y), _cv2.FONT_HERSHEY_SIMPLEX, 0.45, (248, 214, 106), 1); y += 24
    ba = _overlay["base_age"]
    ba_txt = "no baseline" if ba is None else f"baseline {int(ba)}s"
    _cv2.putText(panel, f"gate: {gate}  ({_overlay['occ']}/{_overlay['voting']} occ)  {ba_txt}",
                 (12, y), _cv2.FONT_HERSHEY_SIMPLEX, 0.48, AMBER, 1); y += 28
    _cv2.line(panel, (10, y), (PW - 10, y), (60, 60, 66), 1); y += 22
    nz = _overlay["node_z"]; nodes = _overlay["nodes"]
    for nid in range(1, 7):
        z = nz.get(str(nid), nz.get(nid))
        info = nodes.get(str(nid), nodes.get(nid, {}))
        active = info.get("status") == "active"
        rssi = info.get("rssi")
        over = z is not None and z > STREAM_K
        col = RED if over else (GREEN if z is not None else GREY)
        label = f"n{nid}"
        _cv2.putText(panel, label, (12, y + 4), _cv2.FONT_HERSHEY_SIMPLEX, 0.55, col if active else GREY, 2)
        # deviation bar (z clamped to ~20 for scale)
        bx0, bx1, by = 48, PW - 90, y
        _cv2.rectangle(panel, (bx0, by - 10), (bx1, by + 4), (45, 45, 50), -1)
        if z is not None:
            frac = max(0.0, min(1.0, z / 20.0))
            _cv2.rectangle(panel, (bx0, by - 10), (int(bx0 + (bx1 - bx0) * frac), by + 4), col, -1)
            # threshold tick at K
            kx = int(bx0 + (bx1 - bx0) * (STREAM_K / 20.0))
            _cv2.line(panel, (kx, by - 12), (kx, by + 6), (230, 230, 230), 1)
        zt = "--" if z is None else f"{z:.1f}"
        _cv2.putText(panel, zt, (bx1 + 6, y + 4), _cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        y += 30
    y += 6
    _cv2.putText(panel, "bar = deviation from empty;", (12, y), _cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREY, 1); y += 18
    _cv2.putText(panel, "white tick = person threshold", (12, y), _cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREY, 1)

    composite = np.hstack([f, panel])
    ok, buf = _cv2.imencode(".jpg", composite, [_cv2.IMWRITE_JPEG_QUALITY, 72])
    if ok:
        with _frame_lock:
            _latest_jpeg[0] = buf.tobytes()


def get_gate():
    try:
        with urllib.request.urlopen(BASE + "/api/v1/presence/baseline/status", timeout=4) as r:
            return json.load(r).get("gate") or {}
    except Exception:
        return {}


def get_nodes():
    try:
        with urllib.request.urlopen(BASE + "/api/v1/nodes", timeout=4) as r:
            return {str(n["node_id"]): {"status": n.get("status"), "rssi": n.get("rssi_dbm")}
                    for n in json.load(r).get("nodes", [])}
    except Exception:
        return {}


def get_latest():
    try:
        with urllib.request.urlopen(BASE + "/api/v1/sensing/latest", timeout=4) as r:
            return json.load(r)
    except Exception:
        return {}


class Recorder:
    """Bounded, class-budgeted CSI recorder driven by camera state.

    budgets: max recorded seconds per class. gaps: min seconds between the end
    of one segment and the start of the next, per class — duty-cycling so a
    60-hour empty weekend yields many short, temporally-diverse segments
    instead of one disk-filling stream (~8.4 GB/h raw rate).
    """
    def __init__(self, enabled, budgets=None, gaps=None, seg_secs=REC_SEG_SECS, min_free_gb=50.0):
        self.enabled=enabled; self.active_id=None; self.seg_start=0.0
        self.budget={"empty":0.0,"occupied":0.0}; self.cur_class=None
        self.max_budget=budgets or {"empty":REC_BUDGET_SECS,"occupied":REC_BUDGET_SECS}
        self.gap=gaps or {"empty":0.0,"occupied":0.0}
        self.seg_secs=seg_secs
        self.last_end={"empty":-1e12,"occupied":-1e12}
        self.min_free_gb=min_free_gb          # hard disk floor — never record below this
        self.rec_path="/home/davestanyer/.local/share/ruview/data/recordings"
        self._last_disk_warn=0.0

    def _free_gb(self):
        try:
            return shutil.disk_usage(self.rec_path).free / 1e9
        except Exception:
            return 1e9  # if we can't tell, don't block

    def disk_ok(self):
        free=self._free_gb()
        if free < self.min_free_gb:
            now=time.monotonic()
            if now-self._last_disk_warn > 120:
                self._last_disk_warn=now
                log(f"DISK FLOOR: {free:.0f} GB free < {self.min_free_gb:.0f} GB — recording PAUSED "
                    f"(corpus preserved; free space or raise --min-free-gb to resume)")
            return False
        return True
    def _start(self, cls):
        ts=time.strftime("%Y%m%d_%H%M%S")
        rid=f"auto_{cls}_{ts}"
        try: post("/api/v1/recording/start", {"id":rid}); self.active_id=rid; self.seg_start=time.monotonic(); self.cur_class=cls
        except Exception as e: log(f"record start failed: {e}"); self.active_id=None
        else: log(f"REC start [{cls}] -> {rid} (budget used {self.budget[cls]:.0f}/{self.max_budget[cls]:.0f}s)")
    def _stop(self):
        if not self.active_id: return
        dur=time.monotonic()-self.seg_start
        try: post("/api/v1/recording/stop")
        except Exception as e: log(f"record stop failed: {e}")
        if self.cur_class:
            self.budget[self.cur_class]+=dur
            self.last_end[self.cur_class]=time.monotonic()
        log(f"REC stop  [{self.cur_class}] {self.active_id} +{dur:.0f}s")
        self.active_id=None; self.cur_class=None
    def update(self, stable_state):
        """stable_state: 'empty' | 'occupied' | None (unstable/guard)."""
        if not self.enabled: return
        now=time.monotonic()
        # Rotate FIRST, then decide — computing `want` before the rotation stop
        # let a just-rotated class restart instantly (cur_class was still set),
        # bypassing the duty-cycle gap entirely (found 2026-07-03: continuous
        # 5-min empty segments instead of one per 45 min).
        if self.active_id and (now-self.seg_start)>=self.seg_secs:
            self._stop()
        want=None
        if stable_state in ("empty","occupied") and self.budget[stable_state] < self.max_budget[stable_state]:
            # duty-cycle: respect min gap since this class's last segment ended
            # (unless we're already mid-segment on this class)
            if self.cur_class==stable_state or (now-self.last_end[stable_state])>=self.gap[stable_state]:
                want=stable_state
        if self.active_id and (self.cur_class!=want):
            self._stop()
        # Hard disk floor: don't START a new segment when free space is low
        # (an in-progress segment finishes normally, bounded by seg_secs).
        if want and not self.active_id and self.disk_ok():
            self._start(want)
    def close(self): self._stop()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--no-baseline-refresh", action="store_true",
                    help="observe only: never trigger presence baseline captures")
    ap.add_argument("--empty-budget", type=float, default=REC_BUDGET_SECS,
                    help="max recorded seconds of empty CSI (default %(default)s)")
    ap.add_argument("--occ-budget", type=float, default=REC_BUDGET_SECS,
                    help="max recorded seconds of occupied CSI (default %(default)s)")
    ap.add_argument("--empty-gap", type=float, default=0.0,
                    help="min seconds between empty segments (duty-cycle for long runs)")
    ap.add_argument("--occ-gap", type=float, default=0.0,
                    help="min seconds between occupied segments")
    ap.add_argument("--seg-secs", type=float, default=REC_SEG_SECS,
                    help="rotate recording file every N seconds (default %(default)s)")
    ap.add_argument("--stream-port", type=int, default=8091,
                    help="serve paired camera+verdict MJPEG view on this port (0=off)")
    ap.add_argument("--collect-pose", action="store_true",
                    help="log camera 17-keypoint COCO poses (epoch-stamped) to the "
                         "pose_corpus for later CSI pairing / skeleton training (ADR-079)")
    ap.add_argument("--min-free-gb", type=float, default=50.0,
                    help="hard disk floor: pause recording when free space drops below "
                         "this (GB). Corpus is preserved; never fills the disk (default %(default)s)")
    args=ap.parse_args()

    lm=PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL), running_mode=RunningMode.IMAGE, num_poses=1))
    # The USB webcam re-enumerates to a different /dev/videoN index whenever USB
    # is disturbed (reflashing, replug) — happened 3x on 2026-07-07/08. Auto-scan
    # for the first index that yields real frames instead of hardcoding 0.
    cap=None
    for idx in range(0,6):
        c=cv2.VideoCapture(idx, cv2.CAP_V4L2)
        c.set(cv2.CAP_PROP_FRAME_WIDTH,640); c.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        ok=False
        for _ in range(6):
            r,f=c.read()
            if r and f is not None and f.mean()>1: ok=True; break
            time.sleep(0.1)
        if ok:
            cap=c; log(f"camera found at /dev/video{idx}"); break
        c.release()
    if cap is None:
        log("FATAL: no working camera found on /dev/video0..5"); sys.exit(2)
    for _ in range(8): cap.read(); time.sleep(0.1)

    rec=Recorder(enabled=not args.no_record,
                 budgets={"empty":args.empty_budget,"occupied":args.occ_budget},
                 gaps={"empty":args.empty_gap,"occupied":args.occ_gap},
                 seg_secs=args.seg_secs, min_free_gb=args.min_free_gb)
    labels_f=OUT.joinpath("labels.jsonl").open("a")   # camera-truth timeline for post-hoc slicing
    pose_f=None
    if args.collect_pose:
        POSE_DIR.mkdir(parents=True, exist_ok=True)
        pose_f=POSE_DIR.joinpath(f"pose_{time.strftime('%Y%m%d')}.jsonl").open("a")
        log(f"pose collection ON -> {POSE_DIR} (epoch-stamped COCO-17, pair with CSI offline)")
    last_label=0.0
    last_overlay=0.0      # monotonic time of last paired-view overlay refresh
    last_baseline=-1e12   # monotonic time of last baseline-capture trigger
    last_seen=None          # monotonic time of last person detection
    state=None; state_since=time.monotonic(); last_change=time.monotonic()
    last_score=0.0
    confusion=Counter()   # (camera_truth, ruview_presence)
    running=[True]
    def stop(*_): running[0]=False
    signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)
    t0=time.monotonic()
    log(f"oracle up. record={'off' if args.no_record else 'on'}. Watching...")

    if args.stream_port:
        try:
            _start_stream_server(args.stream_port)
            log(f"paired view: http://127.0.0.1:{args.stream_port}/")
        except Exception as e:
            log(f"stream server failed to start: {e}")

    next_detect=0.0
    while running[0]:
        if args.minutes and (time.monotonic()-t0)>args.minutes*60: break
        now=time.monotonic()
        if now>=next_detect:
            next_detect=now+1.0/DETECT_FPS
            ok,frame=cap.read()
            if ok:
                res=lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))
                lms=None
                if res.pose_landmarks:
                    vis=float(np.mean([l.visibility for l in res.pose_landmarks[0]]))
                    if vis>=VIS_MIN: last_seen=now; lms=res.pose_landmarks[0]
                # metric 3D skeleton (world landmarks, hip-centred metres) for the /3d view
                present_now = last_seen is not None and now-last_seen<=OCC_HOLD
                wl = res.pose_world_landmarks[0] if getattr(res,"pose_world_landmarks",None) else None
                with _frame_lock:
                    _pose3d["present"] = bool(present_now and wl is not None)
                    _pose3d["kp"] = ([[round(wl[m].x,3),round(wl[m].y,3),round(wl[m].z,3)] for m in MP_TO_COCO]
                                     if (present_now and wl is not None) else None)
                if args.stream_port:
                    _overlay["cam"]="person" if (last_seen is not None and now-last_seen<=OCC_HOLD) else "empty"
                    _publish_frame(cv2, np, frame, lms)
                # paired-pose corpus: epoch-stamped COCO-17 keypoints when a person
                # is present. CSI is recorded in parallel (auto_occupied_*, timestamped);
                # align offline by epoch -> CSI frame timestamp -> training sample.
                if pose_f is not None and lms is not None:
                    kps=[[round(lms[m].x,4), round(lms[m].y,4), round(lms[m].z,4),
                          round(lms[m].visibility,3)] for m in MP_TO_COCO]
                    pose_f.write(json.dumps({"epoch":time.time(),"n_persons":1,
                                             "keypoints":kps,"kp_format":"coco17_xyzv"})+"\n")
                    pose_f.flush()
        # asymmetric hysteresis label (fast to occupied, slow to empty)
        inst=None
        if last_seen is not None:
            gap=now-last_seen
            if gap<=OCC_HOLD:      inst="occupied"
            elif gap>=EMPTY_CONFIRM: inst="empty"
            # else: in the dead-band -> keep current state (inst stays None)
        if inst and inst!=state:
            state=inst; state_since=now; last_change=now
            log(f"camera -> {state}")
            if state == "occupied":
                trigger_c3po_presence_greeting()
        stable = state if (state and (now-state_since)>=STABLE_SECS and (now-last_change)>=GUARD_SECS) else None
        rec.update(stable)
        # R3: camera-confirmed empty -> refresh the presence gate's baseline.
        # This is the seam ADR-135 was missing ("no reliable mechanism to
        # verify room emptiness from CSI alone").
        if not args.no_baseline_refresh and stable=="empty" and (now-last_baseline)>=BASELINE_REFRESH_SECS:
            last_baseline=now
            try:
                post("/api/v1/presence/baseline/capture", {"duration_secs": BASELINE_CAPTURE_SECS})
                log(f"presence baseline capture triggered ({BASELINE_CAPTURE_SECS:.0f}s)")
            except Exception as e:
                log(f"baseline capture failed (server may predate R3): {e}")
        # camera-truth timeline (wall-clock) so any recording can be sliced by ground truth
        if now-last_label>=2.0:
            last_label=now
            labels_f.write(json.dumps({"epoch":time.time(),"camera":state,"stable":stable})+"\n"); labels_f.flush()
        # scoring
        if now-last_score>=SCORE_POLL and stable:
            last_score=now
            try:
                p=get_presence()
                confusion[(stable, bool(p))]+=1
                # live tallies
                emp_tot=confusion[("empty",True)]+confusion[("empty",False)]
                occ_tot=confusion[("occupied",True)]+confusion[("occupied",False)]
                fp = 100*confusion[("empty",True)]/emp_tot if emp_tot else float('nan')
                fn = 100*confusion[("occupied",False)]/occ_tot if occ_tot else float('nan')
                log(f"score cam={stable:8s} ruview_present={bool(p)}  | empty→FP={fp:.0f}% (n={emp_tot})  occupied→FN={fn:.0f}% (n={occ_tot})")
                OUT.joinpath("scoreboard.json").write_text(json.dumps({
                    "confusion":{f"{k[0]}|ruview_present={k[1]}":v for k,v in confusion.items()},
                    "empty_false_positive_pct":fp, "occupied_false_negative_pct":fn,
                    "empty_n":emp_tot, "occupied_n":occ_tot,
                    "record_budget":rec.budget, "updated":time.strftime("%H:%M:%S")}, indent=2))
            except Exception as e:
                log(f"score poll err: {e}")
        # refresh the paired-view overlay's RuView side (~every 1.5s, always)
        if args.stream_port and now-last_overlay>=1.5:
            last_overlay=now
            try:
                g=get_gate()
                latest=get_latest()
                cls=latest.get("classification") or {}
                persons=latest.get("persons") or []
                desk=latest.get("desk_occupancy") or {}
                votes=desk.get("votes") or {}
                pres=cls.get("presence")
                _overlay["ruview"]="PRESENT" if pres else "EMPTY"
                _overlay["gate"]=g.get("verdict","?")
                _overlay["voting"]=g.get("nodes_voting",0)
                _overlay["occ"]=g.get("nodes_occupied",0)
                _overlay["base_age"]=g.get("baseline_age_secs")
                _overlay["node_z"]=g.get("node_z") or {}
                _overlay["nodes"]=get_nodes()
                with _frame_lock:
                    _wifi["present"]=bool(pres)
                    _wifi["motion"]=cls.get("motion_level","?")
                    _wifi["verdict"]=g.get("verdict","?")
                    _wifi["nodes_voting"]=int(g.get("nodes_voting") or 0)
                    _wifi["nodes_occupied"]=int(g.get("nodes_occupied") or 0)
                    _wifi["baseline_stale"]=g.get("baseline_stale")
                    _wifi["persons"]=latest.get("estimated_persons") or len(persons)
                    _wifi["pos"]=persons[0].get("position") if persons else None
                    _wifi["desk_state"]=desk.get("state","?")
                    _wifi["desk_confidence"]=float(desk.get("confidence") or 0.0)
                    _wifi["desk_seated"]=bool(desk.get("seated"))
                    _wifi["model_seated"]=float(votes.get("model_seated") or 0.0)
                    _wifi["model_standing_near"]=float(votes.get("model_standing_near") or 0.0)
                    _wifi["model_away"]=float(votes.get("model_away") or 0.0)
            except Exception:
                pass
        time.sleep(0.05)

    rec.close(); cap.release(); labels_f.close()
    if pose_f is not None: pose_f.close()
    log("=== FINAL ===")
    for k in sorted(confusion): log(f"  camera={k[0]:8s} ruview_present={k[1]}: {confusion[k]}")
    emp_tot=confusion[("empty",True)]+confusion[("empty",False)]
    occ_tot=confusion[("occupied",True)]+confusion[("occupied",False)]
    if emp_tot: log(f"  EMPTY  false-present: {100*confusion[('empty',True)]/emp_tot:.1f}%  (n={emp_tot})")
    if occ_tot: log(f"  OCCUP  missed:        {100*confusion[('occupied',False)]/occ_tot:.1f}%  (n={occ_tot})")
    log(f"  recorded: {rec.budget}")

if __name__=="__main__":
    main()
