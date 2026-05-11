"""
DefectVision/utils/convert.py
==============================
Standalone script — converts KolektorSDD2 preprocessed images + raw masks
into a YOLO-format dataset with correct bounding boxes.

OPTION A: Preprocessing has already run. This script:
  1. Reads preprocessed 640x640 images from:
        artifacts/<latest>/data_preprocessing/processed/images/{train,val}/
  2. Reads raw _GT masks from:
        artifacts/<latest>/data_ingestion/feature_store/{train,test}/
  3. Generates YOLO labels (coords correct for 640x640 via letterbox math)
  4. Augments positives until TARGET_POSITIVE_COUNT is reached
  5. Writes final dataset to:
        artifacts/<latest>/data_conversion/yolo_dataset/

KEY FIXES (matched to notebook KolektorSDD2_fixed.ipynb):
  - Bbox merging   : ALL contours merged into ONE union bbox per image
  - Threshold 127  : not 10
  - Letterbox math : coords generated directly in 640x640 space
  - Oversampling   : augment until TARGET_POSITIVE_COUNT, not fixed multiplier

Run from project root:
    python -m DefectVision.utils.convert
OR directly:
    python DefectVision/utils/convert.py
"""

import os
import sys
import cv2
import glob
import shutil
import yaml
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
import albumentations as A
from tqdm import tqdm
import logging

# ── handle both module and direct-script execution ──────────────────
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from DefectVision.constant.training_pipeline import (
    MASK_SUFFIX,
    CLASS_NAMES,
    CLASS_ID,
    MIN_CONTOUR_AREA,
    NEG_TO_POS_RATIO,
    RANDOM_SEED,
    ARTIFACTS_DIR,
    DATA_INGESTION_DIR_NAME,
    DATA_INGESTION_FEATURE_STORE_DIR,
    DATA_PREPROCESSING_DIR_NAME,
    DATA_PREPROCESSING_PROCESSED_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── output folder names ──────────────────────────────────────────────
DATA_CONVERSION_DIR_NAME = "data_conversion"
YOLO_DATASET_DIR_NAME    = "yolo_dataset"

# Target number of positive samples in train after augmentation
# (notebook used 1000)
TARGET_POSITIVE_COUNT = 1000

VAL_SPLIT_RATIO = 0.20


# =====================================================================
# AUGMENTATION PIPELINES  (matched to notebook)
# =====================================================================

AUG_POSITIVE = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT,p=0.5),
        A.RandomBrightnessContrast(p=0.7),
        A.GaussNoise(p=0.4),
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.4,
        clip=True,
    ),
)

AUG_NEGATIVE = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
])


# =====================================================================
# HELPERS
# =====================================================================

def find_mask(img_stem: str, mask_dir: Path):
    """Find <img_stem>_GT.* inside mask_dir."""
    mask_stem = img_stem + MASK_SUFFIX
    for f in mask_dir.iterdir():
        if f.stem == mask_stem:
            return f
    return None


def read_img(path):
    img = cv2.imread(str(path))
    if img is None:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return img


def write_label(path, lines: list):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        if lines:
            f.write("\n".join(lines) + "\n")


# =====================================================================
# MASK -> YOLO BBOX  (notebook-exact: merge all contours into ONE bbox)
# =====================================================================

def mask_to_yolo(mask_path) -> list:
    """
    Read a _GT mask -> return ONE merged YOLO bbox in 640x640 space.

    Notebook logic:
      - threshold at 127
      - collect all contours with area >= MIN_CONTOUR_AREA
      - merge ALL into one union bounding box
      - apply letterbox transform to get 640x640-correct coords

    Returns [] if no valid defect found.
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        log.warning(f"  Cannot read mask: {mask_path}")
        return []

    # notebook uses threshold 127
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    orig_h, orig_w = binary.shape   # e.g. 630 x 230

    # letterbox params — must match KolektorPreprocessor exactly
    TARGET_W, TARGET_H = 640, 640
    scale    = min(TARGET_W / orig_w, TARGET_H / orig_h)
    new_w    = int(orig_w * scale)
    new_h    = int(orig_h * scale)
    pad_left = (TARGET_W - new_w) // 2
    pad_top  = (TARGET_H - new_h) // 2

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # collect all contour bounding boxes
    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < int(MIN_CONTOUR_AREA):
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        boxes.append([x, y, x + bw, y + bh])

    if not boxes:
        return []

    # merge into ONE union bbox (notebook: "merge all boxes into one")
    x_min = min(b[0] for b in boxes)
    y_min = min(b[1] for b in boxes)
    x_max = max(b[2] for b in boxes)
    y_max = max(b[3] for b in boxes)

    bw = x_max - x_min
    bh = y_max - y_min

    # convert raw-pixel -> 640x640 normalised
    xc_px  = (x_min + bw / 2) * scale
    yc_px  = (y_min + bh / 2) * scale
    w_px   = bw * scale
    h_px   = bh * scale

    xc_tgt = xc_px + pad_left
    yc_tgt = yc_px + pad_top

    cx = max(0.0, min(1.0, xc_tgt / TARGET_W))
    cy = max(0.0, min(1.0, yc_tgt / TARGET_H))
    nw = max(0.001, min(1.0, w_px  / TARGET_W))
    nh = max(0.001, min(1.0, h_px  / TARGET_H))

    return [f"{int(CLASS_ID)} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"]


# =====================================================================
# SCAN FOLDERS
# =====================================================================

def scan_folder(img_dir: Path, mask_dir: Path) -> tuple:
    """
    Match preprocessed images (img_dir) to _GT masks (mask_dir).
    Returns:
        positives : list of (img_path, mask_path)
        negatives : list of img_path
    """
    positives, negatives = [], []

    for f in sorted(img_dir.iterdir()):
        if not f.is_file():
            continue
        if f.stem.endswith(MASK_SUFFIX):
            continue

        mask_f = find_mask(f.stem, mask_dir)

        if mask_f is None:
            negatives.append(str(f))
            continue

        mask = cv2.imread(str(mask_f), cv2.IMREAD_GRAYSCALE)
        if mask is not None and mask.max() > 127:
            positives.append((str(f), str(mask_f)))
        else:
            negatives.append(str(f))

    return positives, negatives


# =====================================================================
# AUGMENTATION HELPERS
# =====================================================================

def augment_positive(img_path, yolo_lines, out_img_dir, out_lbl_dir, name) -> bool:
    img = read_img(img_path)
    if img is None:
        return False

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 \
              else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    bboxes, labels = [], []
    for line in yolo_lines:
        parts  = line.split()
        labels.append(int(parts[0]))
        bbox   = [max(0.0, min(1.0, float(v))) for v in parts[1:]]
        bboxes.append(bbox)

    try:
        res = AUG_POSITIVE(image=img_rgb, bboxes=bboxes, class_labels=labels)
    except Exception as e:
        log.warning(f"  Aug error: {e}")
        return False

    if not res["bboxes"]:
        return False

    cv2.imwrite(
        os.path.join(out_img_dir, name + ".png"),
        cv2.cvtColor(res["image"], cv2.COLOR_RGB2BGR),
    )
    aug_lines = [
        f"{int(c)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
        for c, (cx, cy, bw, bh) in zip(res["class_labels"], res["bboxes"])
    ]
    write_label(os.path.join(out_lbl_dir, name + ".txt"), aug_lines)
    return True


# =====================================================================
# PROCESS ONE SPLIT
# =====================================================================

def process_split(
    positives, negatives,
    out_img_dir, out_lbl_dir,
    name,
    do_augment=False,
    target_positive=TARGET_POSITIVE_COUNT,
):
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    stats = defaultdict(int)
    valid_positives = []

    # ── copy all original positives ───────────────────────────────────
    log.info(f"  [{name}] {len(positives)} positive originals ...")
    for img_path, mask_path in tqdm(positives, desc=f"  {name} pos"):
        lines = mask_to_yolo(mask_path)
        stem  = Path(img_path).stem
        dst_i = os.path.join(out_img_dir, stem + ".png")
        dst_l = os.path.join(out_lbl_dir, stem + ".txt")

        shutil.copy2(img_path, dst_i)

        if lines:
            write_label(dst_l, lines)
            stats["positives"] += 1
            valid_positives.append((img_path, lines))
        else:
            stats["converted_to_neg"] += 1

    # ── copy all original negatives ───────────────────────────────────
    log.info(f"  [{name}] {len(negatives)} negative originals ...")
    for img_path in tqdm(negatives, desc=f"  {name} neg"):
        stem = Path(img_path).stem
        shutil.copy2(img_path, os.path.join(out_img_dir, stem + ".png"))
        # NO label file for negatives
        stats["negatives"] += 1

    # ── oversample positives to target (TRAIN only) ───────────────────
    if do_augment and valid_positives:
        need = max(0, target_positive - stats["positives"])
        log.info(
            f"  [{name}] Augmenting positives: "
            f"{stats['positives']} -> {target_positive} ({need} needed)"
        )
        aug_count = 0
        i = 0
        pbar = tqdm(total=need, desc=f"  {name} augment")

        while aug_count < need:
            for img_path, yolo_lines in valid_positives:
                if aug_count >= need:
                    break
                aug_name = f"aug_{i}_{Path(img_path).stem}"
                ok = augment_positive(
                    img_path, yolo_lines,
                    out_img_dir, out_lbl_dir, aug_name,
                )
                if ok:
                    aug_count += 1
                    pbar.update(1)
                i += 1
        pbar.close()
        stats["augmented"] = aug_count

    # ── cap negatives ─────────────────────────────────────────────────
    total_pos = stats["positives"] + stats.get("augmented", 0)
    if NEG_TO_POS_RATIO is not None and total_pos > 0 and do_augment:
        cap = int(total_pos * NEG_TO_POS_RATIO)
        neg_imgs = [
            f for f in Path(out_img_dir).iterdir()
            if f.is_file()
            and not (Path(out_lbl_dir) / (f.stem + ".txt")).exists()
        ]
        if len(neg_imgs) > cap:
            log.info(f"  [{name}] Capping negatives: {len(neg_imgs)} -> {cap}")
            for f in random.sample(neg_imgs, len(neg_imgs) - cap):
                f.unlink()

    stats["total"] = (
        stats["positives"] + stats.get("augmented", 0)
        + stats["negatives"] + stats["converted_to_neg"]
    )
    return dict(stats)


# =====================================================================
# DATA.YAML
# =====================================================================

def write_data_yaml(yolo_dir: str) -> str:
    data = {
        "path" : os.path.abspath(yolo_dir),
        "train": "images/train",
        "val"  : "images/val",
        "nc"   : len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    path = os.path.join(yolo_dir, "data.yaml")
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    log.info(f"  Written: {path}")
    return path


# =====================================================================
# VALIDATE
# =====================================================================

def validate_output(yolo_dir: str):
    issues = defaultdict(list)
    for split in ("train", "val"):
        img_dir = Path(yolo_dir) / "images" / split
        lbl_dir = Path(yolo_dir) / "labels" / split
        if not img_dir.exists():
            continue
        for f in img_dir.iterdir():
            lbl = lbl_dir / (f.stem + ".txt")
            if lbl.exists() and lbl.stat().st_size == 0:
                issues["empty_label"].append(str(lbl))
            if cv2.imread(str(f)) is None:
                issues["corrupt_image"].append(str(f))

    if any(issues.values()):
        log.warning("  Issues found:")
        for k, v in issues.items():
            log.warning(f"    {k}: {len(v)} files")
    else:
        log.info("  Validation passed — no issues found.")


# =====================================================================
# MAIN
# =====================================================================

def main():
    # Find latest preprocessed images folder
    pattern = os.path.join(
        ARTIFACTS_DIR, "*",
        DATA_PREPROCESSING_DIR_NAME,
        DATA_PREPROCESSING_PROCESSED_DIR,
        "images",
    )
    candidates = sorted(glob.glob(pattern))

    if not candidates:
        log.error(
            f"No preprocessed images found.\n"
            f"Searched pattern: {pattern}\n"
            f"Run DataPreprocessing first."
        )
        sys.exit(1)

    preprocessed_img_path = Path(candidates[-1])
    timestamp_dir         = preprocessed_img_path.parents[2]  # artifacts/<ts>

    feature_store_path = (
        timestamp_dir / DATA_INGESTION_DIR_NAME / DATA_INGESTION_FEATURE_STORE_DIR
    )
    yolo_dir = str(
        timestamp_dir / DATA_CONVERSION_DIR_NAME / YOLO_DATASET_DIR_NAME
    )

    log.info("=" * 60)
    log.info("  convert.py — KolektorSDD2 -> YOLO  [notebook-matched]")
    log.info("=" * 60)
    log.info(f"  preprocessed images : {preprocessed_img_path}")
    log.info(f"  feature_store masks : {feature_store_path}")
    log.info(f"  yolo output         : {yolo_dir}")

    # Map preprocessed splits to feature_store mask splits
    SPLIT_MAP = {"train": "train", "val": "test"}

    all_pos, all_neg = {}, {}

    log.info("\n[1/5] Scanning ...")
    for pp_split, fs_split in SPLIT_MAP.items():
        img_dir  = preprocessed_img_path / pp_split
        mask_dir = feature_store_path    / fs_split

        if not img_dir.exists():
            log.warning(f"  preprocessed/{pp_split}/ not found — skipping")
            all_pos[pp_split] = []
            all_neg[pp_split] = []
            continue

        if not mask_dir.exists():
            log.warning(f"  feature_store/{fs_split}/ not found — all treated as negatives")
            all_pos[pp_split] = []
            all_neg[pp_split] = [str(f) for f in sorted(img_dir.iterdir()) if f.is_file()]
            continue

        pos, neg = scan_folder(img_dir, mask_dir)
        all_pos[pp_split] = pos
        all_neg[pp_split] = neg
        log.info(f"  {pp_split}: {len(pos)} positives, {len(neg)} negatives")

    tr_pos = all_pos.get("train", [])
    tr_neg = all_neg.get("train", [])
    va_pos = all_pos.get("val",   [])
    va_neg = all_neg.get("val",   [])

    if not tr_pos:
        raise ValueError(
            "No positive images found in preprocessed/train/.\n"
            "Check that _GT masks exist in feature_store/train/ "
            f"and that mask pixel values exceed 127.\n"
            f"Searched: {feature_store_path / 'train'}"
        )

    log.info("\n[2/5] Converting train ...")
    tr_stats = process_split(
        tr_pos, tr_neg,
        os.path.join(yolo_dir, "images", "train"),
        os.path.join(yolo_dir, "labels", "train"),
        "TRAIN", do_augment=True,
        target_positive=TARGET_POSITIVE_COUNT,
    )

    log.info("\n[3/5] Converting val ...")
    va_stats = process_split(
        va_pos, va_neg,
        os.path.join(yolo_dir, "images", "val"),
        os.path.join(yolo_dir, "labels", "val"),
        "VAL", do_augment=False,
    )

    log.info("\n[4/5] Writing data.yaml ...")
    yaml_path = write_data_yaml(yolo_dir)

    log.info("\n[5/5] Validating ...")
    validate_output(yolo_dir)

    tp   = tr_stats["positives"] + tr_stats.get("augmented", 0)
    vp   = va_stats["positives"]

    log.info("\n" + "=" * 60)
    log.info("  SUMMARY")
    log.info("=" * 60)
    log.info(
        f"\n  TRAIN"
        f"\n    Original positives : {tr_stats['positives']}"
        f"\n    Augmented          : {tr_stats.get('augmented', 0)}"
        f"\n    Total positives    : {tp}"
        f"\n    Negatives kept     : {tr_stats['negatives']}"
        f"\n    Total images       : {tr_stats['total']}"
    )
    log.info(
        f"\n  VAL"
        f"\n    Positives          : {vp}"
        f"\n    Negatives          : {va_stats['negatives']}"
        f"\n    Total images       : {va_stats['total']}"
    )
    log.info(f"\n  data.yaml -> {yaml_path}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()