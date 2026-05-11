"""
DefectVision/utils/inspect.py
==============================
Utility that inspects the raw dataset after data ingestion.

Workflow position:
    constant/training_pipeline.py
        -> entity/config_entity.py       (DataIngestionConfig)
        -> entity/artifact_entity.py     (DataIngestionArtifact)
        -> component/data_ingestion.py   (produces DataIngestionArtifact)
        -> [THIS FILE]                   (reads DataIngestionArtifact)

How to call (from app.py or a notebook):
    from DefectVision.utils.inspect import DatasetInspector

    # after DataIngestion component has run and returned an artifact:
    artifact = DataIngestionArtifact(
        data_zip_file_path="...",
        feature_store_path="artifacts/.../data_ingestion/feature_store"
    )
    inspector = DatasetInspector(artifact)
    report = inspector.run()

Or run directly as a script (reads paths from config_entity):
    python -m DefectVision.utils.inspect
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from DefectVision.entity.artifacts_entity import DataIngestionArtifact
from DefectVision.entity.config_entity import DataIngestionConfig
from DefectVision.constant.training_pipeline import MASK_SUFFIX


# ── return type so callers can use the numbers programmatically ───────────────
@dataclass
class SplitInspectionResult:
    split_name:     str
    total_images:   int
    total_masks:    int
    positives:      int
    negatives:      int
    no_mask:        int
    image_sizes:    dict          # {(h, w): count}
    mask_min:       Optional[int]
    mask_max:       Optional[int]
    mask_mean:      Optional[float]
    neg_pos_ratio:  float


@dataclass
class InspectionReport:
    train: Optional[SplitInspectionResult]
    test:  Optional[SplitInspectionResult]
    feature_store_path: str


class DatasetInspector:
    """
    Inspects the raw KolektorSDD2 dataset that lives inside
    the feature_store_path produced by DataIngestion component.

    Expected folder structure inside feature_store_path:
        feature_store/
        ├── train/
        │   ├── 10000          <- image (extension-less or .png)
        │   ├── 10000_GT       <- mask
        │   ├── 10001
        │   ├── 10001_GT
        │   └── ...
        └── test/              <- optional, may not exist
            ├── ...
            └── ...
    """

    def __init__(self, artifact: DataIngestionArtifact):
        """
        Parameters
        ----------
        artifact : DataIngestionArtifact
            The artifact returned by component/data_ingestion.py.
            We use artifact.feature_store_path to locate the raw images.
        """
        self.artifact = artifact
        self.feature_store = Path(artifact.feature_store_path)

    # ── public entry point ────────────────────────────────────────────────────
    def run(self, verbose: bool = True) -> InspectionReport:
        """
        Inspect train/ and test/ splits inside the feature store.

        Parameters
        ----------
        verbose : bool
            Print the report to stdout.

        Returns
        -------
        InspectionReport
            Dataclass with results for both splits.
        """
        train_dir = self.feature_store / "train"
        test_dir  = self.feature_store / "test"

        if verbose:
            print("=" * 54)
            print("  DefectVision — raw dataset inspection")
            print(f"  feature_store : {self.feature_store}")
            print("=" * 54)

        train_result = self._inspect_split(train_dir, "TRAIN", verbose)
        test_result  = self._inspect_split(test_dir,  "TEST",  verbose)

        if verbose:
            print("\n" + "=" * 54)
            print("  Inspection complete.")
            print("=" * 54)

        return InspectionReport(
            train=train_result,
            test=test_result,
            feature_store_path=str(self.feature_store),
        )

    # ── internal: inspect one split ───────────────────────────────────────────
    def _inspect_split(
        self,
        split_dir: Path,
        split_name: str,
        verbose: bool,
    ) -> Optional[SplitInspectionResult]:

        if not split_dir.exists():
            if verbose:
                print(f"\n  [{split_name}] folder not found: {split_dir}")
                print(f"  [{split_name}] skipping.")
            return None

        all_files = sorted(split_dir.iterdir())
        images    = [f for f in all_files if f.is_file() and not f.stem.endswith(MASK_SUFFIX)]
        masks     = [f for f in all_files if f.is_file() and     f.stem.endswith(MASK_SUFFIX)]

        positives     = 0
        negatives     = 0
        no_mask       = 0
        sizes         = defaultdict(int)
        mask_max_vals = []

        for img_f in images:
            # find paired mask (any extension)
            mask_stem = img_f.stem + MASK_SUFFIX
            mask_f    = None
            for candidate in split_dir.iterdir():
                if candidate.stem == mask_stem:
                    mask_f = candidate
                    break

            # record image dimensions
            img = cv2.imread(str(img_f), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                sizes[img.shape] += 1

            if mask_f is None:
                no_mask   += 1
                negatives += 1
                continue

            mask = cv2.imread(str(mask_f), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                no_mask   += 1
                negatives += 1
                continue

            mv = int(mask.max())
            mask_max_vals.append(mv)
            if mv > 10:
                positives += 1
            else:
                negatives += 1

        ratio = negatives / positives if positives else float("inf")

        arr       = np.array(mask_max_vals) if mask_max_vals else None
        mask_min  = int(arr.min())          if arr is not None else None
        mask_max  = int(arr.max())          if arr is not None else None
        mask_mean = float(arr.mean())       if arr is not None else None

        result = SplitInspectionResult(
            split_name    = split_name,
            total_images  = len(images),
            total_masks   = len(masks),
            positives     = positives,
            negatives     = negatives,
            no_mask       = no_mask,
            image_sizes   = dict(sizes),
            mask_min      = mask_min,
            mask_max      = mask_max,
            mask_mean     = mask_mean,
            neg_pos_ratio = ratio,
        )

        if verbose:
            self._print_result(result, split_dir)

        return result

    # ── pretty printer ────────────────────────────────────────────────────────
    @staticmethod
    def _print_result(r: SplitInspectionResult, split_dir: Path):
        print(f"\n{'─'*54}")
        print(f"  Split : {r.split_name}   ({split_dir})")
        print(f"{'─'*54}")
        print(f"  Total images     : {r.total_images}")
        print(f"  Mask files found : {r.total_masks}")
        print(f"  Positives        : {r.positives}  (mask has defect pixels)")
        print(f"  Negatives        : {r.negatives}  (all-black mask or no mask)")
        print(f"  No mask found    : {r.no_mask}")

        if r.image_sizes:
            print("  Image sizes      :")
            for (h, w), cnt in sorted(r.image_sizes.items()):
                print(f"      {h}x{w}  x{cnt} images")

        if r.mask_max is not None:
            print(
                f"  Mask pixel max   : "
                f"min={r.mask_min}  max={r.mask_max}  mean={r.mask_mean:.1f}"
            )

        ratio_str = f"{r.neg_pos_ratio:.1f}:1" if r.neg_pos_ratio != float("inf") else "inf (no positives)"
        print(f"  Neg:Pos ratio    : {ratio_str}")

        print("  Sample files     :")
        for f in list(Path(str(r.split_name)).parent.iterdir()
                       if False else [])[:5]:
            print(f"      {f.name}")


# ── standalone entry point ────────────────────────────────────────────────────
# Usage: python -m DefectVision.utils.inspect
# Builds config from config_entity (same as the component does)
# so you can run this without having run the full pipeline first —
# as long as the feature_store folder already exists on disk.

# def main():
#     from DefectVision.entity.config_entity import DataIngestionConfig

#     config = DataIngestionConfig()

#     # Build a minimal artifact pointing at the feature store
#     # (mirrors what DataIngestion component returns after downloading)
#     artifact = DataIngestionArtifact(
#         data_zip_file_path=os.path.join(
#             config.data_ingestion_dir, "KolektorSDD2.zip"
#         ),
#         feature_store_path=config.feature_store_file_path,
#     )

#     inspector = DatasetInspector(artifact)
#     report    = inspector.run(verbose=True)

#     return report


def main():
    import glob

    # Find the most recently created data_ingestion folder
    # instead of making a new timestamped config
    base = "artifacts"
    pattern = os.path.join(base, "*", "data_ingestion", "feature_store")
    candidates = sorted(glob.glob(pattern))

    if not candidates:
        print("No feature_store found. Run data ingestion first.")
        return

    # Use the most recent one
    feature_store_path = candidates[-1]
    print(f"Using: {feature_store_path}")

    artifact = DataIngestionArtifact(
        data_zip_file_path=os.path.join(
            os.path.dirname(feature_store_path), "KolektorSDD2.zip"
        ),
        feature_store_path=feature_store_path,
    )

    inspector = DatasetInspector(artifact)
    report = inspector.run(verbose=True)
    return report

if __name__ == "__main__":
    main()