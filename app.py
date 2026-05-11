"""
app.py  (place at project root)
================================
Surface Defect Detection — Gradio web app.

Uses best_yolo11s.pt trained in Colab to detect defects in uploaded images.

Setup (local):
    1. Download best_yolo11s.pt from Google Drive
    2. Place it at:  models/best.pt
    3. pip install -r requirements.txt
    4. python app.py
    5. Open http://localhost:7860

Deploy to Railway:
    See README_DEPLOY.md
"""

import os
import cv2
import numpy as np
import gradio as gr
from pathlib import Path
from ultralytics import YOLO

# ── Model path ────────────────────────────────────────────────────────
MODEL_PATH     = os.environ.get("MODEL_PATH", "models/best.pt")
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.25"))
IMG_SIZE       = 640

# ── Load model once at startup ────────────────────────────────────────
print(f"Loading model: {MODEL_PATH}")
if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}\n"
        "Download best_yolo11s.pt from Google Drive and place at models/best.pt"
    )
model = YOLO(MODEL_PATH)
print("Model ready.")


# ── Inference ─────────────────────────────────────────────────────────
def detect_defects(image: np.ndarray, conf_threshold: float):
    """
    Run YOLO11 inference on an uploaded image.

    Parameters
    ----------
    image          : np.ndarray  RGB image from Gradio
    conf_threshold : float       confidence cutoff from slider

    Returns
    -------
    annotated : np.ndarray  image with bboxes drawn
    summary   : str         detection summary text
    """
    if image is None:
        return None, "No image uploaded."

    # Gradio sends RGB — convert to BGR for YOLO
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    results = model.predict(
        source  = img_bgr,
        conf    = conf_threshold,
        imgsz   = IMG_SIZE,
        verbose = False,
    )

    result     = results[0]
    boxes      = result.boxes
    img_output = img_bgr.copy()

    if len(boxes) == 0:
        summary = "No defects detected."
    else:
        summary = f"{len(boxes)} defect(s) detected:\n\n"

        for i, box in enumerate(boxes):
            conf_val        = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            w, h            = x2 - x1, y2 - y1

            # Red bounding box
            cv2.rectangle(img_output, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Label with confidence
            label      = f"defect {conf_val:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            label_y    = max(y1 - 10, label_size[1] + 5)
            cv2.rectangle(
                img_output,
                (x1, label_y - label_size[1] - 5),
                (x1 + label_size[0] + 5, label_y + 3),
                (0, 0, 255), -1,
            )
            cv2.putText(
                img_output, label, (x1 + 2, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )

            summary += (
                f"  Defect {i+1}:\n"
                f"    Confidence : {conf_val:.1%}\n"
                f"    Position   : x={x1}, y={y1}\n"
                f"    Size       : {w} x {h} px\n\n"
            )

    annotated = cv2.cvtColor(img_output, cv2.COLOR_BGR2RGB)
    return annotated, summary


# ── Gradio UI ─────────────────────────────────────────────────────────
with gr.Blocks(title="Surface Defect Detection", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # Surface Defect Detection
    Upload a surface image to detect manufacturing defects using YOLO11s.
    Trained on KolektorSDD2 — detects cracks and surface defects.
    **Model:** YOLO11s &nbsp;|&nbsp; **mAP50:** 0.6741 &nbsp;|&nbsp; **Precision:** 0.7464
    """)

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label   = "Upload Surface Image",
                type    = "numpy",
                sources = ["upload", "clipboard"],
            )
            conf_slider = gr.Slider(
                minimum = 0.10,
                maximum = 0.90,
                value   = CONF_THRESHOLD,
                step    = 0.05,
                label   = "Confidence Threshold",
                info    = "Lower = more detections. Higher = more certain detections only.",
            )
            detect_btn = gr.Button("Detect Defects", variant="primary")

        with gr.Column():
            output_image = gr.Image(label="Result")
            output_text  = gr.Textbox(label="Detection Summary", lines=10)

    detect_btn.click(
        fn      = detect_defects,
        inputs  = [input_image, conf_slider],
        outputs = [output_image, output_text],
    )


# ── Launch ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name = "0.0.0.0",
        server_port = int(os.environ.get("PORT", 7860)),
        share       = False,
    )







