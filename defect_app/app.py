"""
app.py  —  DefectVision  —  Surface Defect Detection
======================================================
Pure FastAPI + HTML/CSS/JS — zero Gradio dependency.
Local:  python app.py  →  http://localhost:7860
Railway: Push to GitHub → connect on railway.app → done.
"""

import os, io, cv2, base64, time
import numpy as np
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import uvicorn

MODEL_PATH     = os.environ.get("MODEL_PATH", "models/best.pt")
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.25"))
IMG_SIZE       = 640
PORT           = int(os.environ.get("PORT", 7860))

print(f"[INIT] Loading model: {MODEL_PATH}")
if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print("[INIT] Model ready.")

app = FastAPI(title="DefectVision")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


def run_inference(img_array: np.ndarray, conf: float) -> dict:
    t0      = time.time()
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    h, w    = img_bgr.shape[:2]
    results  = model.predict(source=img_bgr, conf=conf, imgsz=IMG_SIZE, verbose=False)
    result   = results[0]
    boxes    = result.boxes
    img_out  = img_bgr.copy()
    elapsed  = (time.time() - t0) * 1000
    n        = len(boxes)
    detections = []

    for i, box in enumerate(boxes):
        c            = float(box.conf[0])
        x1,y1,x2,y2 = [int(v) for v in box.xyxy[0].tolist()]
        bw, bh       = x2-x1, y2-y1
        color        = (0, 165, 255)
        cv2.rectangle(img_out, (x1,y1), (x2,y2), color, 2)
        tick = 12
        for cx,cy,dx,dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(img_out,(cx,cy),(cx+dx*tick,cy),color,3)
            cv2.line(img_out,(cx,cy),(cx,cy+dy*tick),color,3)
        label = f"#{i+1} {c:.0%}"
        lw,lh = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.55,2)[0]
        ly    = max(y1-6, lh+6)
        cv2.rectangle(img_out,(x1,ly-lh-6),(x1+lw+10,ly+2),color,-1)
        cv2.putText(img_out,label,(x1+5,ly-1),cv2.FONT_HERSHEY_SIMPLEX,0.55,(15,15,15),2)
        detections.append({"id":i+1,"conf":round(c,4),
                           "x1":x1,"y1":y1,"x2":x2,"y2":y2,
                           "w":bw,"h":bh,"area":bw*bh})

    vc = (0,80,255) if n>0 else (50,200,50)
    cv2.putText(img_out,"FAIL" if n>0 else "PASS",(w-110,h-14),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,vc,2)

    buf = io.BytesIO()
    Image.fromarray(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        "image_b64" : f"data:image/png;base64,{b64}",
        "n_defects" : n,
        "status"    : "DEFECT DETECTED" if n>0 else "SURFACE CLEAR",
        "verdict"   : "FAIL" if n>0 else "PASS",
        "infer_ms"  : round(elapsed,1),
        "img_w"     : w, "img_h": h,
        "detections": detections,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), conf: float = Form(CONF_THRESHOLD)):
    try:
        data      = await file.read()
        nparr     = np.frombuffer(data, np.uint8)
        img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_array is None:
            return JSONResponse({"error": "Could not decode image. Use PNG or JPG."}, status_code=400)
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        return JSONResponse(run_inference(img_rgb, conf))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>DefectVision</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Syne:wght@400;700;800&display=swap" rel="stylesheet"/>
<style>
:root{
  --bg0:#0e0f11;--bg1:#14161a;--bg2:#1c1f25;--bg3:#252930;
  --border:#2e3340;--amber:#f59e0b;--amber2:#fbbf24;
  --green:#22c55e;--red:#ef4444;--text:#e2e8f0;--muted:#64748b;
  --mono:'Share Tech Mono',monospace;--sans:'Syne',sans-serif;
}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg0);color:var(--text);font-family:var(--sans);min-height:100vh;}
.container{max-width:1100px;margin:0 auto;padding:28px 20px;}
.header{border-bottom:1px solid var(--border);padding-bottom:20px;margin-bottom:20px;}
.header h1{font-size:2.2rem;font-weight:800;letter-spacing:-.02em;}
.header h1 em{color:var(--amber);font-style:normal;}
.header p{font-family:var(--mono);font-size:.72rem;color:var(--muted);margin-top:5px;letter-spacing:.07em;}
.stats{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:20px;}
.chip{background:var(--bg2);border:1px solid var(--border);border-radius:6px;padding:6px 13px;font-family:var(--mono);font-size:.68rem;color:var(--muted);}
.chip b{color:var(--amber);}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;}
@media(max-width:700px){.grid{grid-template-columns:1fr;}}
.panel{background:var(--bg1);border:1px solid var(--border);border-radius:10px;padding:20px;}
.sec-label{font-family:var(--mono);font-size:.62rem;color:var(--amber);letter-spacing:.14em;text-transform:uppercase;margin-bottom:12px;display:flex;align-items:center;gap:7px;}
.sec-label::before{content:'';width:6px;height:6px;background:var(--amber);border-radius:50%;animation:blink 2s infinite;flex-shrink:0;}
@keyframes blink{0%,100%{opacity:1;}50%{opacity:.2;}}
.dropzone{border:1.5px dashed var(--border);border-radius:8px;background:var(--bg2);
          min-height:260px;display:flex;flex-direction:column;align-items:center;
          justify-content:center;cursor:pointer;transition:border-color .2s,background .2s;
          overflow:hidden;user-select:none;}
.dropzone:hover,.dropzone.dragover{border-color:var(--amber);background:var(--bg3);}
.dz-icon{font-size:2.5rem;margin-bottom:10px;opacity:.4;}
.dz-text{font-family:var(--mono);font-size:.72rem;color:var(--muted);}
.dz-hint{font-family:var(--mono);font-size:.6rem;color:var(--muted);opacity:.6;margin-top:5px;}
#preview{width:100%;height:100%;object-fit:contain;display:none;border-radius:8px;}
.slider-wrap{margin-top:16px;}
.slider-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;}
.slider-head span{font-family:var(--mono);font-size:.65rem;color:var(--muted);letter-spacing:.07em;}
.slider-head strong{font-family:var(--mono);font-size:.75rem;color:var(--amber);}
input[type=range]{width:100%;accent-color:var(--amber);}
.btn{background:var(--amber);color:#0a0b0d;border:none;border-radius:8px;font-family:var(--sans);
     font-size:.9rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;padding:14px;
     width:100%;cursor:pointer;transition:all .18s;margin-top:14px;}
.btn:hover{background:var(--amber2);transform:translateY(-1px);box-shadow:0 6px 20px rgba(245,158,11,.3);}
.btn:active{transform:translateY(0);}
.btn:disabled{opacity:.5;cursor:not-allowed;transform:none;}
.out-img-wrap{position:relative;min-height:260px;background:var(--bg2);border-radius:8px;
              overflow:hidden;display:flex;align-items:center;justify-content:center;}
.out-img-wrap::after{content:'';position:absolute;top:-2px;left:0;right:0;height:2px;
                     background:linear-gradient(90deg,transparent,rgba(245,158,11,.7),transparent);
                     animation:scan 3s linear infinite;pointer-events:none;}
@keyframes scan{0%{top:-2px;}100%{top:100%;}}
#resultImg{width:100%;border-radius:8px;display:none;}
.out-placeholder{font-family:var(--mono);font-size:.7rem;color:var(--muted);text-align:center;opacity:.5;}
.report{background:var(--bg2);border:1px solid var(--border);border-radius:8px;padding:14px;
        font-family:var(--mono);font-size:.74rem;line-height:1.7;color:var(--text);
        min-height:160px;margin-top:12px;white-space:pre-wrap;word-break:break-word;}
.verdict{display:inline-block;padding:4px 14px;border-radius:5px;font-family:var(--mono);
         font-size:.72rem;font-weight:700;letter-spacing:.1em;margin-bottom:10px;}
.verdict.pass{background:rgba(34,197,94,.15);color:var(--green);border:1px solid rgba(34,197,94,.3);}
.verdict.fail{background:rgba(239,68,68,.15);color:var(--red);border:1px solid rgba(239,68,68,.3);}
.verdict.waiting{background:var(--bg3);color:var(--muted);border:1px solid var(--border);}
.spinner{display:none;width:28px;height:28px;border:3px solid var(--border);
         border-top-color:var(--amber);border-radius:50%;
         animation:spin .7s linear infinite;margin:0 auto;}
@keyframes spin{to{transform:rotate(360deg);}}
.footer{margin-top:24px;padding-top:14px;border-top:1px solid var(--border);
        display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;}
.footer p{font-family:var(--mono);font-size:.62rem;color:var(--muted);}
.badges{display:flex;gap:6px;}
.badge{background:var(--bg2);border:1px solid var(--border);border-radius:4px;
       padding:3px 9px;font-family:var(--mono);font-size:.6rem;color:var(--amber);}
</style>
</head>
<body>
<div class="container">

  <div class="header">
    <h1>Defect<em>Vision</em></h1>
    <p>SURFACE DEFECT DETECTION &nbsp;·&nbsp; YOLO11s &nbsp;·&nbsp; KolektorSDD2 &nbsp;·&nbsp; v1.0</p>
  </div>

  <div class="stats">
    <div class="chip">MODEL &nbsp;<b>YOLO11s</b></div>
    <div class="chip">mAP50 &nbsp;<b>0.6741</b></div>
    <div class="chip">PRECISION &nbsp;<b>0.7464</b></div>
    <div class="chip">RECALL &nbsp;<b>0.6422</b></div>
    <div class="chip">CLASSES &nbsp;<b>1 — defect</b></div>
    <div class="chip">INPUT &nbsp;<b>640 × 640</b></div>
  </div>

  <div class="grid">
    <div class="panel">
      <div class="sec-label">Input</div>

      <!-- fileInput lives OUTSIDE dropzone — no nesting, no double-trigger -->
      <input type="file" id="fileInput" accept="image/png,image/jpeg,image/bmp,image/webp"
             style="position:fixed;top:-9999px;left:-9999px;opacity:0;"/>

      <div class="dropzone" id="dropzone">
        <div class="dz-icon" id="dzIcon">⬡</div>
        <div class="dz-text" id="dzText">Drop image or click to upload</div>
        <div class="dz-hint">PNG · JPG · BMP supported</div>
        <img id="preview"/>
      </div>

      <div class="slider-wrap">
        <div class="slider-head">
          <span>CONFIDENCE THRESHOLD</span>
          <strong id="confVal">25%</strong>
        </div>
        <input type="range" id="confSlider" min="10" max="90" value="25" step="5"/>
        <div style="display:flex;justify-content:space-between;font-family:var(--mono);font-size:.58rem;color:var(--muted);margin-top:4px;">
          <span>↓ more detections</span><span>fewer, more certain ↑</span>
        </div>
      </div>

      <button class="btn" id="detectBtn" disabled>⬡ &nbsp;Run Detection</button>
    </div>

    <div class="panel">
      <div class="sec-label">Analysis Output</div>
      <div id="verdictBadge" class="verdict waiting">── AWAITING INPUT ──</div>
      <div class="out-img-wrap">
        <div class="out-placeholder" id="outPlaceholder">Upload an image and click Run Detection</div>
        <div class="spinner" id="spinner"></div>
        <img id="resultImg" alt="detection result"/>
      </div>
      <div class="report" id="report">── AWAITING INPUT ──

Upload a surface image and click Run Detection
to begin defect analysis.</div>
    </div>
  </div>

  <div class="footer">
    <p>DefectVision &nbsp;·&nbsp; YOLO11s &nbsp;·&nbsp; KolektorSDD2 &nbsp;·&nbsp; 2026</p>
    <div class="badges">
      <span class="badge">PyTorch</span>
      <span class="badge">Ultralytics</span>
      <span class="badge">FastAPI</span>
    </div>
  </div>
</div>

<script>
const fileInput  = document.getElementById('fileInput');
const dropzone   = document.getElementById('dropzone');
const preview    = document.getElementById('preview');
const dzIcon     = document.getElementById('dzIcon');
const dzText     = document.getElementById('dzText');
const confSlider = document.getElementById('confSlider');
const confVal    = document.getElementById('confVal');
const detectBtn  = document.getElementById('detectBtn');
const resultImg  = document.getElementById('resultImg');
const report     = document.getElementById('report');
const spinner    = document.getElementById('spinner');
const outPlace   = document.getElementById('outPlaceholder');
const verdict    = document.getElementById('verdictBadge');

let selectedFile = null;

// Slider
confSlider.addEventListener('input', () => {
  confVal.textContent = confSlider.value + '%';
});

// Click dropzone → open file dialog
// fileInput is fixed off-screen so clicking it never bubbles back into dropzone
dropzone.addEventListener('click', (e) => {
  e.stopPropagation();
  fileInput.click();
});

// File chosen via dialog
fileInput.addEventListener('change', () => {
  const f = fileInput.files[0];
  if (f) {
    loadFile(f);
    fileInput.value = ''; // reset so same file can be re-selected
  }
});

// Drag over
dropzone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropzone.classList.add('dragover');
});
dropzone.addEventListener('dragleave', (e) => {
  if (!dropzone.contains(e.relatedTarget))
    dropzone.classList.remove('dragover');
});
// Drop
dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.classList.remove('dragover');
  const f = e.dataTransfer.files[0];
  if (f) loadFile(f);
});

function loadFile(file) {
  selectedFile = file;
  const url = URL.createObjectURL(file);
  preview.onload = () => URL.revokeObjectURL(url);
  preview.src           = url;
  preview.style.display = 'block';
  dzIcon.style.display  = 'none';
  dzText.style.display  = 'none';
  detectBtn.disabled    = false;
}

// Run Detection
detectBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  detectBtn.disabled      = true;
  spinner.style.display   = 'block';
  outPlace.style.display  = 'none';
  resultImg.style.display = 'none';
  verdict.className       = 'verdict waiting';
  verdict.textContent     = '── PROCESSING ──';
  report.textContent      = 'Running YOLO11s inference...';

  const conf = parseInt(confSlider.value) / 100;
  const fd   = new FormData();
  fd.append('file', selectedFile);
  fd.append('conf', String(conf));

  try {
    const res  = await fetch('/predict', { method: 'POST', body: fd });
    const data = await res.json();

    spinner.style.display = 'none';

    if (data.error) {
      report.textContent  = 'Error: ' + data.error;
      verdict.className   = 'verdict waiting';
      verdict.textContent = '── ERROR ──';
      return;
    }

    resultImg.src           = data.image_b64;
    resultImg.style.display = 'block';

    verdict.className   = data.verdict === 'PASS' ? 'verdict pass' : 'verdict fail';
    verdict.textContent = data.verdict === 'PASS'
      ? '✓  PASS — SURFACE CLEAR'
      : '✕  FAIL — DEFECT DETECTED';

    let txt = 'STATUS  ›  ' + data.status + '\n';
    txt += '─'.repeat(38) + '\n';
    txt += 'INFERENCE TIME  : ' + data.infer_ms + ' ms\n';
    txt += 'IMAGE SIZE      : ' + data.img_w + ' × ' + data.img_h + ' px\n';
    txt += 'CONFIDENCE THRS : ' + Math.round(conf * 100) + '%\n';
    txt += 'DEFECTS FOUND   : ' + data.n_defects + '\n';

    if (data.n_defects === 0) {
      txt += '\nNo surface defects detected.\nSurface meets quality standards.';
    } else {
      txt += '\nDEFECT DETAILS\n' + '─'.repeat(38);
      data.detections.forEach(d => {
        txt += '\n\n  Defect #' + d.id;
        txt += '\n  ├ Confidence : ' + (d.conf * 100).toFixed(1) + '%';
        txt += '\n  ├ Centroid   : (' + Math.round((d.x1+d.x2)/2) + ', ' + Math.round((d.y1+d.y2)/2) + ')';
        txt += '\n  ├ Bbox size  : ' + d.w + ' × ' + d.h + ' px';
        txt += '\n  └ Bbox area  : ' + d.area.toLocaleString() + ' px²';
      });
    }
    report.textContent = txt;

  } catch (err) {
    spinner.style.display = 'none';
    report.textContent    = 'Request failed: ' + err.message;
    verdict.className     = 'verdict waiting';
    verdict.textContent   = '── ERROR ──';
  } finally {
    detectBtn.disabled = false;
  }
});
</script>
</body>
</html>"""


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=False)