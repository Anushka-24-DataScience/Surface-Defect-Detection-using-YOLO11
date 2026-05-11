"""
app.py  —  Surface Defect Detection
=====================================
Industrial-grade UI with custom CSS theme.
YOLO11s trained on KolektorSDD2.

Local run:
    pip install -r requirements.txt
    python app.py

Railway deploy:
    See README_DEPLOY.md
"""

import os
import cv2
import numpy as np
import gradio as gr
from pathlib import Path
from ultralytics import YOLO
import time

# ── Config ────────────────────────────────────────────────────────────
MODEL_PATH     = os.environ.get("MODEL_PATH", "models/best.pt")
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.25"))
IMG_SIZE       = 640

# ── Load model ────────────────────────────────────────────────────────
print(f"[INIT] Loading model: {MODEL_PATH}")
if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}\n"
        "Place best_yolo11s.pt at models/best.pt"
    )
model = YOLO(MODEL_PATH)
print("[INIT] Model ready.")


# ── Inference ─────────────────────────────────────────────────────────
def detect_defects(image: np.ndarray, conf_threshold: float):
    if image is None:
        return None, "── AWAITING INPUT ──\n\nUpload an image to begin analysis."

    t_start  = time.time()
    img_bgr  = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w     = img_bgr.shape[:2]

    results  = model.predict(
        source  = img_bgr,
        conf    = conf_threshold,
        imgsz   = IMG_SIZE,
        verbose = False,
    )

    result     = results[0]
    boxes      = result.boxes
    img_out    = img_bgr.copy()
    t_infer    = (time.time() - t_start) * 1000
    n_defects  = len(boxes)

    STATUS     = "DEFECT DETECTED" if n_defects > 0 else "SURFACE CLEAR"
    VERDICT    = "FAIL" if n_defects > 0 else "PASS"

    for i, box in enumerate(boxes):
        conf_val        = float(box.conf[0])
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        bw, bh          = x2 - x1, y2 - y1

        # Amber bounding box (matches UI accent)
        color = (0, 165, 255)   # BGR amber
        cv2.rectangle(img_out, (x1, y1), (x2, y2), color, 2)

        # Corner tick marks — industrial style
        tick = 12
        for (cx, cy, dx, dy) in [
            (x1, y1, 1,  1), (x2, y1, -1,  1),
            (x1, y2, 1, -1), (x2, y2, -1, -1),
        ]:
            cv2.line(img_out, (cx, cy), (cx + dx * tick, cy), color, 3)
            cv2.line(img_out, (cx, cy), (cx, cy + dy * tick), color, 3)

        # Label background
        label = f"#{i+1}  {conf_val:.0%}"
        lw, lh = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
        ly = max(y1 - 6, lh + 6)
        cv2.rectangle(img_out, (x1, ly - lh - 6), (x1 + lw + 10, ly + 2),
                      color, -1)
        cv2.putText(img_out, label, (x1 + 5, ly - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (15, 15, 15), 2)

    # Watermark verdict
    vcolor = (0, 100, 255) if n_defects > 0 else (60, 180, 60)
    cv2.putText(img_out, VERDICT,
                (w - 110, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, vcolor, 2)

    annotated = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)

    # ── Text report ───────────────────────────────────────────────────
    lines = [
        f"STATUS  ›  {STATUS}",
        f"{'─' * 38}",
        f"INFERENCE TIME  : {t_infer:.1f} ms",
        f"IMAGE SIZE      : {w} × {h} px",
        f"CONFIDENCE THRS : {conf_threshold:.0%}",
        f"DEFECTS FOUND   : {n_defects}",
        "",
    ]

    if n_defects == 0:
        lines.append("No surface defects detected above threshold.")
        lines.append("Surface meets quality standards.")
    else:
        lines.append("DEFECT DETAILS")
        lines.append(f"{'─' * 38}")
        for i, box in enumerate(boxes):
            conf_val        = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            bw, bh          = x2 - x1, y2 - y1
            cx              = (x1 + x2) // 2
            cy              = (y1 + y2) // 2
            lines += [
                f"",
                f"  Defect #{i+1}",
                f"  ├ Confidence : {conf_val:.1%}",
                f"  ├ Centroid   : ({cx}, {cy})",
                f"  ├ Bbox size  : {bw} × {bh} px",
                f"  └ Bbox area  : {bw*bh:,} px²",
            ]

    return annotated, "\n".join(lines)


# ── Custom CSS — Industrial Precision theme ───────────────────────────
CSS = """
/* ── Google Fonts ────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Syne:wght@400;600;700;800&display=swap');

/* ── Root variables ──────────────────────────────────────────────── */
:root {
    --bg0      : #0e0f11;
    --bg1      : #14161a;
    --bg2      : #1c1f25;
    --bg3      : #252930;
    --border   : #2e3340;
    --amber    : #f59e0b;
    --amber-dim: #78500a;
    --green    : #22c55e;
    --red      : #ef4444;
    --text     : #e2e8f0;
    --muted    : #64748b;
    --mono     : 'Share Tech Mono', monospace;
    --sans     : 'Syne', sans-serif;
}

/* ── Global reset ────────────────────────────────────────────────── */
* { box-sizing: border-box; }

body, .gradio-container {
    background: var(--bg0) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
}

/* ── Main container ──────────────────────────────────────────────── */
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 24px !important;
}

/* ── Header ──────────────────────────────────────────────────────── */
#header {
    border-bottom: 1px solid var(--border);
    padding-bottom: 20px;
    margin-bottom: 28px;
}
#header h1 {
    font-family: var(--sans) !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em;
    color: var(--text) !important;
    margin: 0 0 6px !important;
}
#header h1 span { color: var(--amber); }
#header p {
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
    color: var(--muted) !important;
    margin: 0 !important;
    letter-spacing: 0.06em;
}

/* ── Stats bar ───────────────────────────────────────────────────── */
.stat-bar {
    display: flex;
    gap: 12px;
    margin-bottom: 24px;
    flex-wrap: wrap;
}
.stat-chip {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px 16px;
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--muted);
    letter-spacing: 0.05em;
}
.stat-chip span {
    color: var(--amber);
    font-weight: 700;
}

/* ── Panels ──────────────────────────────────────────────────────── */
.panel {
    background: var(--bg1) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 20px !important;
    position: relative;
}
.panel-label {
    font-family: var(--mono);
    font-size: 0.65rem;
    color: var(--amber);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.panel-label::before {
    content: '';
    display: inline-block;
    width: 6px; height: 6px;
    background: var(--amber);
    border-radius: 50%;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
}

/* ── Upload area ─────────────────────────────────────────────────── */
.upload-area {
    border: 1.5px dashed var(--border) !important;
    border-radius: 8px !important;
    background: var(--bg2) !important;
    min-height: 280px !important;
    transition: border-color 0.2s;
}
.upload-area:hover {
    border-color: var(--amber) !important;
}

/* ── Slider ──────────────────────────────────────────────────────── */
input[type="range"] {
    accent-color: var(--amber) !important;
}
.slider-label {
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
    color: var(--muted) !important;
    letter-spacing: 0.05em;
}

/* ── Detect button ───────────────────────────────────────────────── */
#detect-btn {
    background: var(--amber) !important;
    color: #0e0f11 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: var(--sans) !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 14px 32px !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    margin-top: 16px !important;
}
#detect-btn:hover {
    background: #fbbf24 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(245, 158, 11, 0.35) !important;
}
#detect-btn:active {
    transform: translateY(0) !important;
}

/* ── Output image ────────────────────────────────────────────────── */
.output-image {
    border-radius: 8px !important;
    overflow: hidden !important;
    border: 1px solid var(--border) !important;
    background: var(--bg2) !important;
    min-height: 280px !important;
}

/* ── Report textbox ──────────────────────────────────────────────── */
textarea, .report-box {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
    line-height: 1.65 !important;
    padding: 16px !important;
    resize: none !important;
}
textarea:focus {
    border-color: var(--amber) !important;
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(245, 158, 11, 0.15) !important;
}

/* ── Labels ──────────────────────────────────────────────────────── */
label span, .gr-block-label {
    font-family: var(--mono) !important;
    font-size: 0.65rem !important;
    color: var(--muted) !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}

/* ── Footer ──────────────────────────────────────────────────────── */
#footer {
    margin-top: 28px;
    padding-top: 16px;
    border-top: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
}
#footer p {
    font-family: var(--mono) !important;
    font-size: 0.68rem !important;
    color: var(--muted) !important;
    margin: 0 !important;
}
#footer .badge {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 4px 10px;
    font-family: var(--mono);
    font-size: 0.65rem;
    color: var(--amber);
}

/* ── Scan line animation on output ───────────────────────────────── */
.scan-wrap {
    position: relative;
    overflow: hidden;
    border-radius: 8px;
}
.scan-wrap::after {
    content: '';
    position: absolute;
    top: -100%;
    left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg,
        transparent 0%,
        rgba(245,158,11,0.6) 50%,
        transparent 100%);
    animation: scan 3s linear infinite;
    pointer-events: none;
}
@keyframes scan {
    0%   { top: -2px; }
    100% { top: 100%; }
}

/* ── Gradio internal overrides ───────────────────────────────────── */
.gr-panel, .gr-box {
    background: var(--bg1) !important;
    border-color: var(--border) !important;
}
footer { display: none !important; }
.progress-bar { background: var(--amber) !important; }

/* ── Mobile ──────────────────────────────────────────────────────── */
@media (max-width: 640px) {
    #header h1 { font-size: 1.5rem !important; }
    .stat-bar   { gap: 8px; }
    .stat-chip  { font-size: 0.65rem; padding: 6px 12px; }
}
"""


# ── Build UI ──────────────────────────────────────────────────────────
with gr.Blocks(
    title = "DefectVision — Surface Defect Detection",
    css   = CSS,
    theme = gr.themes.Base(
        primary_hue   = "amber",
        neutral_hue   = "slate",
        font          = gr.themes.GoogleFont("Syne"),
        font_mono     = gr.themes.GoogleFont("Share Tech Mono"),
    ),
) as demo:

    # ── Header ──────────────────────────────────────────────────────
    gr.HTML("""
    <div id="header">
        <h1>Defect<span>Vision</span></h1>
        <p>SURFACE DEFECT DETECTION SYSTEM  ·  YOLO11s  ·  KolektorSDD2  ·  v1.0</p>
    </div>
    """)

    # ── Stats bar ────────────────────────────────────────────────────
    gr.HTML("""
    <div class="stat-bar">
        <div class="stat-chip">MODEL &nbsp;<span>YOLO11s</span></div>
        <div class="stat-chip">mAP50 &nbsp;<span>0.6741</span></div>
        <div class="stat-chip">PRECISION &nbsp;<span>0.7464</span></div>
        <div class="stat-chip">RECALL &nbsp;<span>0.6422</span></div>
        <div class="stat-chip">CLASSES &nbsp;<span>1 (defect)</span></div>
        <div class="stat-chip">INPUT SIZE &nbsp;<span>640 × 640</span></div>
    </div>
    """)

    # ── Main layout ──────────────────────────────────────────────────
    with gr.Row(equal_height=True):

        # ── Left: input panel ────────────────────────────────────────
        with gr.Column(scale=1):
            gr.HTML('<div class="panel-label">INPUT</div>')

            input_image = gr.Image(
                label       = "Surface Image",
                type        = "numpy",
                sources     = ["upload", "clipboard"],
                elem_classes= ["upload-area"],
                show_label  = True,
                height      = 320,
            )

            conf_slider = gr.Slider(
                minimum     = 0.10,
                maximum     = 0.90,
                value       = CONF_THRESHOLD,
                step        = 0.05,
                label       = "Confidence Threshold",
                info        = "↓ Lower → more detections   ↑ Higher → fewer, more certain",
                elem_classes= ["slider-label"],
            )

            detect_btn = gr.Button(
                "⬡  Run Detection",
                variant    = "primary",
                elem_id    = "detect-btn",
            )

        # ── Right: output panel ──────────────────────────────────────
        with gr.Column(scale=1):
            gr.HTML('<div class="panel-label">ANALYSIS OUTPUT</div>')

            output_image = gr.Image(
                label       = "Detection Result",
                type        = "numpy",
                interactive = False,
                elem_classes= ["output-image", "scan-wrap"],
                height      = 320,
            )

            output_text = gr.Textbox(
                label       = "Inspection Report",
                lines       = 10,
                max_lines   = 20,
                interactive = False,
                placeholder = "── AWAITING INPUT ──\n\nUpload an image and click Run Detection.",
                elem_classes= ["report-box"],
            )

    # ── Button click ─────────────────────────────────────────────────
    detect_btn.click(
        fn      = detect_defects,
        inputs  = [input_image, conf_slider],
        outputs = [output_image, output_text],
    )

    # ── Examples ─────────────────────────────────────────────────────
    gr.HTML("""
    <div style="margin-top:24px; font-family:'Share Tech Mono',monospace;
                font-size:0.68rem; color:#64748b; letter-spacing:0.08em;">
        USAGE  ›  Upload a KolektorSDD2-style surface image (PNG/JPG).
        Adjust confidence threshold to tune sensitivity.
        Red boxes = detected defects. PASS/FAIL verdict overlaid on image.
    </div>
    """)

    # ── Footer ───────────────────────────────────────────────────────
    gr.HTML("""
    <div id="footer">
        <p>DefectVision · YOLO11s · Trained on KolektorSDD2 · 2026</p>
        <div style="display:flex;gap:8px;">
            <span class="badge">PyTorch</span>
            <span class="badge">Ultralytics</span>
            <span class="badge">Gradio</span>
        </div>
    </div>
    """)


# ── Launch ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name = "0.0.0.0",
        server_port = int(os.environ.get("PORT", 7860)),
        share       = False,
    )