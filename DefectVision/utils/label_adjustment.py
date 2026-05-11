"""
DefectVision/utils/label_adjustment.py
=======================================
Utility: adjusts YOLO label coordinates after letterbox padding.

WHY THIS IS NEEDED
------------------
utils/convert.py creates YOLO labels by normalising bbox coords against
the RAW image size (e.g. 230x630).

When data_preprocessing.py letterboxes those same images to 640x640,
it adds gray padding (114) around the content. The bbox coords in the
.txt files still reference the raw image space — they are now WRONG
relative to the new 640x640 image because the content region has shifted.

This utility fixes those coords in-place.

WHAT IT DOES (from notebook adjust_for_letterbox function)
----------------------------------------------------------
For each label file:
  1. Convert YOLO normalised (cx, cy, w, h) -> pixel coords in content region
     pixel_x = cx * new_content_w
     pixel_y = cy * new_content_h
  2. Add letterbox padding offset
     pixel_x_in_640 = pixel_x + pad_left
     pixel_y_in_640 = pixel_y + pad_top
  3. Re-normalise against 640x640
     new_cx = pixel_x_in_640 / 640
     new_cy = pixel_y_in_640 / 640

WHERE IT FITS IN THE WORKFLOW
------------------------------
    constant/training_pipeline.py
        -> entity/config_entity.py      (DataPreprocessingConfig)
        -> entity/artifact_entity.py    (DataPreprocessingArtifact)
        -> component/data_preprocessing.py   (calls LabelAdjustment internally)
            -> utils/label_adjustment.py     (THIS FILE — math only)
        -> pipeline/training_pipeline.py
        -> app.py

This file contains ONLY the label adjustment math.
No pipeline logic, no config imports.
"""

import os
from pathlib import Path
from tqdm import tqdm
import logging

log = logging.getLogger(__name__)


class LabelAdjustment:
    """
    Adjusts YOLO label coordinates from raw image space
    to letterboxed image space.

    Parameters
    ----------
    original_size : (width, height)  raw image size before preprocessing
                    e.g. (230, 630) for KolektorSDD2
    target_size   : (width, height)  size images were padded to
                    e.g. (640, 640)
    """

    def __init__(self, original_size: tuple, target_size: tuple):
        orig_w, orig_h   = original_size
        tgt_w,  tgt_h   = target_size

        # Compute the same scale + padding that letterbox_resize used
        scale            = min(tgt_w / orig_w, tgt_h / orig_h)
        self.new_w       = int(orig_w * scale)
        self.new_h       = int(orig_h * scale)
        self.pad_left    = (tgt_w - self.new_w) // 2
        self.pad_top     = (tgt_h - self.new_h) // 2
        self.tgt_w       = tgt_w
        self.tgt_h       = tgt_h

    def adjust_line(self, line: str) -> str:
        """
        Adjust one YOLO annotation line.

        Input  : "0 0.923729 0.146930 0.152542 0.293860"
                  (normalised against original raw image)
        Output : "0 0.412345 0.312456 0.068100 0.131700"
                  (normalised against 640x640 letterboxed image)
        """
        parts = line.strip().split()
        if len(parts) < 5:
            return line   # malformed — return unchanged

        cls = parts[0]   # already an int string e.g. "0"
        xc, yc, w, h = map(float, parts[1:])

        # Step 1: convert from raw normalised -> pixel in content region
        xc_px = xc * self.new_w
        yc_px = yc * self.new_h
        w_px  =  w * self.new_w
        h_px  =  h * self.new_h

        # Step 2: shift by letterbox padding
        xc_tgt = xc_px + self.pad_left
        yc_tgt = yc_px + self.pad_top

        # Step 3: re-normalise to target image size, clamp to [0,1]
        xc_f = max(0.0, min(1.0, xc_tgt / self.tgt_w))
        yc_f = max(0.0, min(1.0, yc_tgt / self.tgt_h))
        w_f  = max(0.001, min(1.0, w_px  / self.tgt_w))
        h_f  = max(0.001, min(1.0, h_px  / self.tgt_h))

        return f"{cls} {xc_f:.6f} {yc_f:.6f} {w_f:.6f} {h_f:.6f}"

    def adjust_file(self, label_path: str):
        """Adjust all lines in one .txt label file in-place."""
        label_path = Path(label_path)
        content = label_path.read_text().strip()

        if not content:
            return   # empty file = negative sample, nothing to adjust

        adjusted = [self.adjust_line(line) for line in content.splitlines()]
        label_path.write_text("\n".join(adjusted) + "\n")

    def adjust_dir(self, label_dir: str) -> int:
        """
        Adjust all .txt label files in a directory in-place.

        Parameters
        ----------
        label_dir : str  path to folder containing .txt YOLO label files

        Returns
        -------
        int  number of files adjusted
        """
        label_dir = Path(label_dir)
        label_files = list(label_dir.glob("*.txt"))

        if not label_files:
            log.warning(f"  No .txt files found in: {label_dir}")
            return 0

        for lbl_f in tqdm(label_files, desc=f"  adjusting [{label_dir.name}]"):
            self.adjust_file(str(lbl_f))

        log.info(f"  {len(label_files)} label files adjusted in {label_dir}")
        return len(label_files)