"""
DefectVision/component/data_preprocessing.py
=============================================
Stage 3 — preprocesses RAW images from feature_store.

Workflow position:
    constant/training_pipeline.py
        -> entity/config_entity.py       (DataPreprocessingConfig)
        -> entity/artifact_entity.py     (DataPreprocessingArtifact)
        -> [THIS FILE]                   (image processing only)
        -> utils/convert.py              (runs next — reads preprocessed images)

What this component does:
    1. Reads raw images from feature_store/train/ and feature_store/test/
    2. Applies: denoise -> CLAHE -> sharpen -> letterbox 640x640 -> BGR
    3. Saves to preprocessed/images/train/ and preprocessed/images/val/
    4. Returns DataPreprocessingArtifact

What this component does NOT do:
    - Does NOT touch labels  (convert.py creates labels from masks)
    - Does NOT augment       (convert.py augments preprocessed images)
    - Does NOT read masks    (convert.py reads masks for label generation)

Why Option A (preprocess first):
    - convert.py augments preprocessed images -> augmented copies are
      also denoised, sharpened, contrast-enhanced (better quality)
    - Labels from masks are normalised against 640x640 directly
      so NO label coordinate adjustment is needed at all
"""

import os
import cv2
import glob
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging

from DefectVision.entity.config_entity   import DataPreprocessingConfig
from DefectVision.entity.artifacts_entity import DataPreprocessingArtifact
from DefectVision.constant.training_pipeline import (
    ARTIFACTS_DIR,
    DATA_INGESTION_DIR_NAME,
    DATA_INGESTION_FEATURE_STORE_DIR,
    DATA_PREPROCESSING_DIR_NAME,
    DATA_PREPROCESSING_PROCESSED_DIR,
    MASK_SUFFIX,
    PREPROCESSING_TARGET_SIZE,
    PREPROCESSING_PAD_VALUE
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =====================================================================
# IMAGE PROCESSOR  (matches notebook KolektorPreprocessor exactly)
# =====================================================================
class KolektorPreprocessor:
    """
    Preprocessing pipeline for KolektorSDD2 grayscale images.

    Steps (from notebook, in order):
        load_image   grayscale read
        denoise      fastNlMeansDenoising h=10, template=7, search=21
        apply_clahe  CLAHE clipLimit=2.0 tileGrid=(8,8)
        sharpen      unsharp mask — GaussianBlur + addWeighted
        resize       letterbox pad to 640x640 with value=114
        to_bgr       grayscale to BGR  (cv2.imwrite needs BGR)
    """

    def __init__(self, target_size: tuple = (640, 640), pad_value: int = 114):
        self.target_size = target_size      # (width, height)
        self.pad_value   = pad_value
        self.clahe       = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Cannot read image: {path}")
        return img

    def denoise(self, img: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoising(img, None, 10, 7, 21)

    def apply_clahe(self, img: np.ndarray) -> np.ndarray:
        return self.clahe.apply(img)

    def sharpen(self, img: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=2)
        return cv2.addWeighted(img, 1.5, blurred, -0.5, 0)

    def resize(self, img: np.ndarray) -> np.ndarray:
        """
        Letterbox resize — keeps aspect ratio, pads remainder with pad_value.

        Example: raw 230x630 image
            scale     = min(640/230, 640/630) = min(2.78, 1.016) = 1.016
            new size  = 234 x 640
            pad_left  = (640-234)//2 = 203
            pad_right = 640-234-203  = 203
            result    = 640x640 with content centred
        """
        h, w               = img.shape
        target_w, target_h = self.target_size
        scale              = min(target_w / w, target_h / h)
        new_w              = int(w * scale)
        new_h              = int(h * scale)
        resized            = cv2.resize(img, (new_w, new_h),
                                        interpolation=cv2.INTER_LINEAR)
        pad_top            = (target_h - new_h) // 2
        pad_bottom         = target_h - new_h - pad_top
        pad_left           = (target_w - new_w) // 2
        pad_right          = target_w - new_w - pad_left
        return cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=self.pad_value,
        )

    def to_bgr(self, img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    def process(self, path: str) -> np.ndarray:
        """Full pipeline. Returns 640x640 BGR image ready for YOLO."""
        img = self.load_image(path)
        img = self.denoise(img)
        img = self.apply_clahe(img)
        img = self.sharpen(img)
        img = self.resize(img)
        img = self.to_bgr(img)
        return img


# =====================================================================
# COMPONENT CLASS  (follows your workflow pattern)
# =====================================================================
class DataPreprocessing:
    """
    Workflow component — reads raw images, preprocesses, saves 640x640 images.

    Called by pipeline/training_pipeline.py.
    Must run BEFORE utils/convert.py.

    Parameters
    ----------
    config : DataPreprocessingConfig
        Built by config_entity — contains all paths and settings.
    """

    def __init__(self, config: DataPreprocessingConfig):
        self.config       = config
        self.preprocessor = KolektorPreprocessor(
            target_size=config.target_size,
            pad_value=config.pad_value,
        )

    def initiate_data_preprocessing(self) -> DataPreprocessingArtifact:
        log.info("=" * 60)
        log.info("  DataPreprocessing — raw images -> 640x640")
        log.info("=" * 60)
        log.info(f"  Input  : {self.config.feature_store_path}")
        log.info(f"  Output : {self.config.preprocessed_dir}")

        preprocessed_dir = Path(self.config.preprocessed_dir)

        # feature_store/train -> preprocessed/images/train
        # feature_store/test  -> preprocessed/images/val
        split_map = {"train": "train", "test": "val"}

        for raw_split, out_split in split_map.items():
            src_dir = Path(self.config.feature_store_path) / raw_split
            dst_dir = preprocessed_dir / "images" / out_split

            if not src_dir.exists():
                log.warning(f"  [{raw_split}] not found — skipping: {src_dir}")
                continue

            dst_dir.mkdir(parents=True, exist_ok=True)

            # skip _GT mask files — only process real images
            images = [
                f for f in sorted(src_dir.iterdir())
                if f.is_file() and not f.stem.endswith(MASK_SUFFIX)
            ]

            log.info(f"  [{raw_split} -> {out_split}] {len(images)} images ...")
            ok = failed = 0

            for img_path in tqdm(images, desc=f"  {out_split}"):
                try:
                    processed = self.preprocessor.process(str(img_path))
                    cv2.imwrite(
                        str(dst_dir / (img_path.stem + ".png")),
                        processed,
                    )
                    ok += 1
                except Exception as e:
                    log.warning(f"  Failed {img_path.name}: {e}")
                    failed += 1

            log.info(f"  [{out_split}] done — ok: {ok}, failed: {failed}")

        artifact = DataPreprocessingArtifact(
            preprocessed_dir = str(preprocessed_dir),
            train_img_dir    = str(preprocessed_dir / "images" / "train"),
            val_img_dir      = str(preprocessed_dir / "images" / "val"),
            target_size      = self.config.target_size,
        )

        log.info("\n  DataPreprocessingArtifact:")
        log.info(f"    train_img_dir : {artifact.train_img_dir}")
        log.info(f"    val_img_dir   : {artifact.val_img_dir}")
        log.info(f"    target_size   : {artifact.target_size}")
        log.info("\n  Done. Run utils/convert.py next.")
        log.info("=" * 60)
        return artifact


# =====================================================================
# STANDALONE ENTRY POINT
# python -m DefectVision.component.data_preprocessing
# =====================================================================
def main():
    import glob

    # find most recent feature_store that already has data on disk
    pattern    = os.path.join(
        ARTIFACTS_DIR, "*",
        DATA_INGESTION_DIR_NAME,
        DATA_INGESTION_FEATURE_STORE_DIR,
    )
    candidates = sorted(glob.glob(pattern))

    if not candidates:
        log.error("No feature_store found. Run data_ingestion first.")
        return

    feature_store = candidates[-1]
    timestamp_dir = Path(feature_store).parents[1]  # artifacts/<timestamp>

    log.info(f"Using feature_store : {feature_store}")
    log.info(f"Using timestamp_dir : {timestamp_dir}")

    # manually build config using the EXISTING timestamp folder
    # so we never create a new timestamp
    config                     = object.__new__(DataPreprocessingConfig)
    config.feature_store_path  = feature_store
    config.preprocessed_dir    = str(
        timestamp_dir
        / DATA_PREPROCESSING_DIR_NAME
        / DATA_PREPROCESSING_PROCESSED_DIR
    )
    config.target_size         = PREPROCESSING_TARGET_SIZE
    config.pad_value           = PREPROCESSING_PAD_VALUE

    comp = DataPreprocessing(config=config)
    comp.initiate_data_preprocessing()


if __name__ == "__main__":
    main()

