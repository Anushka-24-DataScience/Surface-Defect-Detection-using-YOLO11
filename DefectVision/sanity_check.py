"""
sanity_check.py
===============
Run from project root:
    python sanity_check.py

Opens 8 random positive images from your yolo_dataset and draws
the bounding boxes on them. Saves output to sanity_check_output/ folder.
"""

import cv2
import random
from pathlib import Path

# ── UPDATE THIS PATH if your timestamp folder is different ──────────
YOLO_DIR = Path(r"artifacts\05_01_2026_22_50_34\data_conversion\yolo_dataset")
# ────────────────────────────────────────────────────────────────────

IMG_DIR = YOLO_DIR / "images" / "train"
LBL_DIR = YOLO_DIR / "labels" / "train"
OUT_DIR = Path("sanity_check_output")
OUT_DIR.mkdir(exist_ok=True)

# Collect all label files that have content (positives)
positive_labels = [
    f for f in LBL_DIR.glob("*.txt")
    if f.stat().st_size > 0
]

if not positive_labels:
    print("ERROR: No label files with content found!")
    print(f"Checked: {LBL_DIR}")
    print("This means bbox generation failed — check convert.py output.")
    exit(1)

print(f"Found {len(positive_labels)} positive label files")
print(f"Drawing bboxes on 8 random samples ...\n")

samples = random.sample(positive_labels, min(8, len(positive_labels)))

for lbl_path in samples:
    img_path = IMG_DIR / (lbl_path.stem + ".png")

    if not img_path.exists():
        print(f"  SKIP — image not found: {img_path.name}")
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  SKIP — cannot read image: {img_path.name}")
        continue

    h, w = img.shape[:2]
    lines = lbl_path.read_text().strip().splitlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls, cx, cy, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

        # Convert YOLO normalised -> pixel coords
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        # Draw green bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label text
        cv2.putText(
            img, f"defect ({cx:.2f},{cy:.2f})",
            (x1, max(y1 - 8, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
        )

    out_path = OUT_DIR / img_path.name
    cv2.imwrite(str(out_path), img)
    print(f"  Saved: {out_path}  (bbox lines: {len(lines)})")

print(f"\nDone! Open the folder: {OUT_DIR.resolve()}")
print("You should see green rectangles around the defects.")