# """
# DefectVision/components/data_validation.py
# ==========================================
# Stage: Data Validation

# Workflow position:
#     constant/training_pipeline.py
#         -> entity/config_entity.py       (DataValidationConfig)
#         -> entity/artifacts_entity.py    (DataConversionArtifact, DataValidationArtifact)
#         -> [THIS FILE]                   (validates YOLO dataset produced by utils/convert.py)
#         -> pipeline/training_pipeline.py

# Why conversion is NOT a pipeline stage:
#     Data conversion is a one-time utility that depends on the target model format.
#     YOLO needs normalised .txt labels, COCO needs JSON, etc.
#     Keeping it as utils/convert.py makes the pipeline model-agnostic and reusable.
#     The pipeline simply validates whatever YOLO dataset exists on disk.

# What this component does:
#     1. Checks required folder structure (images/train, images/val, labels/train, labels/val)
#     2. Validates every label file — correct format, valid class id, bbox in [0,1]
#     3. Handles negative samples correctly (no label file = valid background image)
#     4. Generates a full dataset statistics report saved to data_validation/report.txt
#     5. Returns DataValidationArtifact(validation_status=True/False)
# """

# import os
# import sys
# import cv2
# from pathlib import Path

# from DefectVision.logger import logging
# from DefectVision.exception import DefectException

# from DefectVision.entity.config_entity import DataValidationConfig
# from DefectVision.entity.artifacts_entity import (
#     DataConversionArtifact,
#     DataValidationArtifact,
# )
# from DefectVision.constant.training_pipeline import CLASS_NAMES


# # =====================================================================
# # DATA VALIDATION COMPONENT
# # =====================================================================
# class DataValidation:
#     """
#     Validates the YOLO dataset produced by utils/convert.py.

#     Parameters
#     ----------
#     data_conversion_artifact : DataConversionArtifact
#         Contains yolo_dataset_path — path to the YOLO dataset root.
#         Must have: images/train, images/val, labels/train, labels/val

#     data_validation_dir : str
#         Where to save validation report and status file.
#         Example: artifacts/<timestamp>/data_validation

#     num_classes : int
#         Number of classes in the dataset.
#         Must match CLASS_NAMES in constant/training_pipeline.py
#         For KolektorSDD2: num_classes = 1 (defect only)
#     """

#     def __init__(
#         self,
#         data_conversion_artifact: DataConversionArtifact,
#         data_validation_dir: str,
#         num_classes: int,
#     ):
#         try:
#             self.artifact         = data_conversion_artifact
#             self.validation_dir   = data_validation_dir
#             self.num_classes      = num_classes
#             self.dataset_path     = Path(self.artifact.yolo_dataset_path)

#             os.makedirs(self.validation_dir, exist_ok=True)
#             self.status_file = os.path.join(self.validation_dir, "status.txt")
#             self.report_file = os.path.join(self.validation_dir, "report.txt")

#         except Exception as e:
#             raise DefectException(e, sys)

#     # ================================================================
#     # 1. VALIDATE ONE LABEL FILE
#     # ================================================================
#     def validate_label_file(self, label_path: Path) -> bool:
#         """
#         Validate one YOLO .txt label file.

#         Rules:
#             - Each line must have exactly 5 values: class cx cy w h
#             - class_id must be integer in [0, num_classes)
#             - cx, cy, w, h must all be floats in [0.0, 1.0]
#             - Empty file is VALID — it means a negative (background) sample

#         Returns True if valid, False if any line fails.
#         """
#         try:
#             content = label_path.read_text().strip()

#             # empty label file = negative sample = valid
#             if not content:
#                 return True

#             for line in content.splitlines():
#                 parts = line.strip().split()

#                 # must have exactly 5 parts: class cx cy w h
#                 if len(parts) != 5:
#                     logging.error(
#                         f"Invalid format in {label_path.name} "
#                         f"(expected 5 values, got {len(parts)}): '{line}'"
#                     )
#                     return False

#                 # class id must be an integer
#                 try:
#                     class_id = int(parts[0])
#                 except ValueError:
#                     logging.error(
#                         f"class_id is not an integer in {label_path.name}: '{parts[0]}'"
#                     )
#                     return False

#                 # class id must be in valid range
#                 if class_id < 0 or class_id >= self.num_classes:
#                     logging.error(
#                         f"class_id {class_id} out of range "
#                         f"[0, {self.num_classes}) in {label_path.name}"
#                     )
#                     return False

#                 # bbox values must be floats in [0, 1]
#                 try:
#                     bbox = [float(v) for v in parts[1:]]
#                 except ValueError:
#                     logging.error(
#                         f"Non-float bbox value in {label_path.name}: {parts[1:]}"
#                     )
#                     return False

#                 for val in bbox:
#                     if val < 0.0 or val > 1.0:
#                         logging.error(
#                             f"BBox value {val:.6f} out of [0,1] "
#                             f"in {label_path.name}: {bbox}"
#                         )
#                         return False

#             return True

#         except Exception as e:
#             raise DefectException(e, sys)

#     # ================================================================
#     # 2. VALIDATE FULL YOLO DATASET
#     # ================================================================
#     def validate_yolo_dataset(self) -> bool:
#         """
#         Full dataset validation:
#             1. Required folder structure exists
#             2. data.yaml exists
#             3. Every image is readable (not corrupt)
#             4. Every label file (where present) is valid YOLO format
#             5. Negative samples (no label file) are correctly accepted

#         Returns True if dataset is valid, False otherwise.
#         """
#         try:
#             logging.info(f"Validating YOLO dataset at: {self.dataset_path}")

#             # ── 1. Folder structure ───────────────────────────────────
#             required_dirs = [
#                 self.dataset_path / "images" / "train",
#                 self.dataset_path / "images" / "val",
#                 self.dataset_path / "labels" / "train",
#                 self.dataset_path / "labels" / "val",
#             ]

#             for d in required_dirs:
#                 if not d.exists():
#                     logging.error(f"Missing required directory: {d}")
#                     return False

#             logging.info("Folder structure check passed")

#             # ── 2. data.yaml ──────────────────────────────────────────
#             yaml_path = self.dataset_path / "data.yaml"
#             if not yaml_path.exists():
#                 logging.error(f"Missing data.yaml at: {yaml_path}")
#                 return False

#             logging.info("data.yaml found")

#             # ── 3. Image + label checks ───────────────────────────────
#             for split in ["train", "val"]:
#                 img_dir = self.dataset_path / "images" / split
#                 lbl_dir = self.dataset_path / "labels" / split

#                 images = [f for f in img_dir.iterdir() if f.is_file()]
#                 logging.info(f"[{split}] checking {len(images)} images ...")

#                 for img_path in images:
#                     label_path = lbl_dir / (img_path.stem + ".txt")

#                     # no label file = negative sample = VALID
#                     # do NOT fail here — this is correct YOLO behaviour
#                     if not label_path.exists():
#                         continue

#                     # corrupt image check
#                     if cv2.imread(str(img_path)) is None:
#                         logging.error(f"Corrupt image: {img_path.name}")
#                         return False

#                     # validate label format
#                     if not self.validate_label_file(label_path):
#                         return False

#                 logging.info(f"[{split}] passed")

#             logging.info("YOLO dataset validation passed")
#             return True

#         except Exception as e:
#             raise DefectException(e, sys)

#     # ================================================================
#     # 3. GENERATE STATISTICS REPORT
#     # ================================================================
#     def generate_dataset_report(self) -> dict:
#         """
#         Generates statistics about the YOLO dataset:
#             - image counts per split
#             - positive / negative counts
#             - total bounding boxes
#             - class distribution
#             - average bbox width and height
#             - corrupt image count
#         """
#         try:
#             stats = {
#                 "yolo_dataset_path"   : str(self.dataset_path),
#                 "num_classes"         : self.num_classes,
#                 "class_names"         : CLASS_NAMES,
#                 "train_images"        : 0,
#                 "val_images"          : 0,
#                 "train_positives"     : 0,
#                 "train_negatives"     : 0,
#                 "val_positives"       : 0,
#                 "val_negatives"       : 0,
#                 "total_bboxes"        : 0,
#                 "empty_label_files"   : 0,
#                 "corrupt_images"      : 0,
#                 "class_distribution"  : {},
#                 "avg_bbox_width"      : 0.0,
#                 "avg_bbox_height"     : 0.0,
#             }

#             total_w = total_h = count_bbox = 0

#             for split in ["train", "val"]:
#                 img_dir = self.dataset_path / "images" / split
#                 lbl_dir = self.dataset_path / "labels" / split

#                 for img_path in img_dir.iterdir():
#                     if not img_path.is_file():
#                         continue

#                     # count images
#                     if split == "train":
#                         stats["train_images"] += 1
#                     else:
#                         stats["val_images"] += 1

#                     # corrupt check
#                     if cv2.imread(str(img_path)) is None:
#                         stats["corrupt_images"] += 1
#                         continue

#                     label_path = lbl_dir / (img_path.stem + ".txt")

#                     # no label file = negative sample
#                     if not label_path.exists():
#                         if split == "train":
#                             stats["train_negatives"] += 1
#                         else:
#                             stats["val_negatives"] += 1
#                         continue

#                     content = label_path.read_text().strip()

#                     # empty label file = negative sample
#                     if not content:
#                         stats["empty_label_files"] += 1
#                         if split == "train":
#                             stats["train_negatives"] += 1
#                         else:
#                             stats["val_negatives"] += 1
#                         continue

#                     # positive sample
#                     if split == "train":
#                         stats["train_positives"] += 1
#                     else:
#                         stats["val_positives"] += 1

#                     for line in content.splitlines():
#                         parts = line.strip().split()
#                         if len(parts) != 5:
#                             continue

#                         class_id = int(parts[0])
#                         w        = float(parts[3])
#                         h        = float(parts[4])

#                         stats["class_distribution"].setdefault(class_id, 0)
#                         stats["class_distribution"][class_id] += 1
#                         stats["total_bboxes"] += 1

#                         total_w    += w
#                         total_h    += h
#                         count_bbox += 1

#             if count_bbox > 0:
#                 stats["avg_bbox_width"]  = round(total_w / count_bbox, 6)
#                 stats["avg_bbox_height"] = round(total_h / count_bbox, 6)

#             # derived stats
#             stats["train_neg_pos_ratio"] = (
#                 round(stats["train_negatives"] / stats["train_positives"], 2)
#                 if stats["train_positives"] > 0 else "inf"
#             )
#             stats["val_neg_pos_ratio"] = (
#                 round(stats["val_negatives"] / stats["val_positives"], 2)
#                 if stats["val_positives"] > 0 else "inf"
#             )

#             return stats

#         except Exception as e:
#             raise DefectException(e, sys)

#     # ================================================================
#     # PIPELINE ENTRY POINT
#     # ================================================================
#     def initiate_data_validation(self) -> DataValidationArtifact:
#         """
#         Runs full validation + generates report.
#         Called by pipeline/training_pipeline.py.
#         """
#         logging.info("=" * 56)
#         logging.info("  DataValidation — YOLO dataset check")
#         logging.info("=" * 56)

#         try:
#             # 1. validate
#             status = self.validate_yolo_dataset()

#             # 2. generate report
#             stats = self.generate_dataset_report()

#             # 3. save report to file
#             with open(self.report_file, "w") as f:
#                 f.write("=" * 56 + "\n")
#                 f.write("  DefectVision — Data Validation Report\n")
#                 f.write("=" * 56 + "\n\n")
#                 for k, v in stats.items():
#                     f.write(f"  {k:<25}: {v}\n")
#                 f.write(f"\n  validation_status     : {status}\n")

#             # 4. save status file
#             with open(self.status_file, "w") as f:
#                 f.write(f"validation_status: {status}\n")

#             # 5. log summary
#             logging.info(
#                 f"\n  TRAIN"
#                 f"\n    images    : {stats['train_images']}"
#                 f"\n    positives : {stats['train_positives']}"
#                 f"\n    negatives : {stats['train_negatives']}"
#                 f"\n    neg:pos   : {stats['train_neg_pos_ratio']}:1"
#                 f"\n  VAL"
#                 f"\n    images    : {stats['val_images']}"
#                 f"\n    positives : {stats['val_positives']}"
#                 f"\n    negatives : {stats['val_negatives']}"
#                 f"\n  LABELS"
#                 f"\n    total bboxes      : {stats['total_bboxes']}"
#                 f"\n    class_distribution: {stats['class_distribution']}"
#                 f"\n    avg bbox width    : {stats['avg_bbox_width']}"
#                 f"\n    avg bbox height   : {stats['avg_bbox_height']}"
#                 f"\n    corrupt images    : {stats['corrupt_images']}"
#             )
#             logging.info(f"  Report saved : {self.report_file}")
#             logging.info(f"  Status       : {status}")
#             logging.info("=" * 56)

#             return DataValidationArtifact(validation_status=status)

#         except Exception as e:
#             raise DefectException(e, sys)


"""
DefectVision/components/data_validation.py
==========================================
Validates the YOLO dataset produced by utils/convert.py.

Why conversion is a utility not a pipeline stage:
    Conversion depends on the target model format (YOLO needs .txt,
    COCO needs JSON, etc). Keeping utils/convert.py separate makes
    the pipeline model-agnostic. The pipeline validates whatever
    YOLO dataset exists on disk.
"""

import os
import sys
import cv2
from pathlib import Path

from DefectVision.logger import logging
from DefectVision.exception import DefectException

from DefectVision.entity.artifacts_entity import (
    DataConversionArtifact,
    DataValidationArtifact,
)
from DefectVision.constant.training_pipeline import (
    CLASS_NAMES,
    DATA_VALIDATION_REQUIRED_DIRS,
    DATA_VALIDATION_STATUS_FILE,
    DATA_VALIDATION_REPORT_FILE,
)


class DataValidation:
    """
    Validates YOLO dataset structure, image integrity, and label format.

    Parameters
    ----------
    data_conversion_artifact : DataConversionArtifact
        Contains yolo_dataset_path pointing to the YOLO dataset root.
    data_validation_dir : str
        Where to save status.txt and report.txt.
        Example: artifacts/<timestamp>/data_validation
    num_classes : int
        Number of classes — must match CLASS_NAMES in constants.
        KolektorSDD2 = 1
    """

    def __init__(
        self,
        data_conversion_artifact: DataConversionArtifact,
        data_validation_dir: str,
        num_classes: int,
    ):
        try:
            self.artifact            = data_conversion_artifact
            self.data_validation_dir = data_validation_dir
            self.num_classes         = num_classes
            self.dataset_path        = Path(self.artifact.yolo_dataset_path)

            os.makedirs(self.data_validation_dir, exist_ok=True)

            self.status_file = os.path.join(
                self.data_validation_dir, DATA_VALIDATION_STATUS_FILE
            )
            self.report_file = os.path.join(
                self.data_validation_dir, DATA_VALIDATION_REPORT_FILE
            )

        except Exception as e:
            raise DefectException(e, sys)

    # ================================================================
    # 1. VALIDATE ONE LABEL FILE
    # ================================================================
    def validate_label_file(self, label_path: Path) -> bool:
        """
        Rules:
            - Empty file = negative sample = VALID
            - Each line: exactly 5 values: class cx cy w h
            - class_id: integer in [0, num_classes)
            - cx cy w h: floats all in [0.0, 1.0]
        """
        try:
            content = label_path.read_text().strip()
            if not content:
                return True   # empty = negative = valid

            for line in content.splitlines():
                parts = line.strip().split()

                if len(parts) != 5:
                    logging.error(
                        f"Invalid format in {label_path.name} "
                        f"(expected 5, got {len(parts)}): '{line}'"
                    )
                    return False

                try:
                    class_id = int(parts[0])
                except ValueError:
                    logging.error(
                        f"class_id not integer in {label_path.name}: '{parts[0]}'"
                    )
                    return False

                if class_id < 0 or class_id >= self.num_classes:
                    logging.error(
                        f"class_id {class_id} out of range "
                        f"[0,{self.num_classes}) in {label_path.name}"
                    )
                    return False

                try:
                    bbox = [float(v) for v in parts[1:]]
                except ValueError:
                    logging.error(
                        f"Non-float bbox in {label_path.name}: {parts[1:]}"
                    )
                    return False

                for val in bbox:
                    if val < 0.0 or val > 1.0:
                        logging.error(
                            f"BBox {val:.6f} out of [0,1] in {label_path.name}"
                        )
                        return False

            return True

        except Exception as e:
            raise DefectException(e, sys)

    # ================================================================
    # 2. VALIDATE FULL YOLO DATASET
    # ================================================================
    def validate_yolo_dataset(self) -> bool:
        try:
            logging.info(f"Validating YOLO dataset: {self.dataset_path}")

            # folder structure
            for rel_dir in DATA_VALIDATION_REQUIRED_DIRS:
                d = self.dataset_path / rel_dir
                if not d.exists():
                    logging.error(f"Missing directory: {d}")
                    return False
            logging.info("Folder structure check passed")

            # data.yaml
            if not (self.dataset_path / "data.yaml").exists():
                logging.error(f"Missing data.yaml at: {self.dataset_path}")
                return False
            logging.info("data.yaml found")

            # image + label checks
            for split in ["train", "val"]:
                img_dir = self.dataset_path / "images" / split
                lbl_dir = self.dataset_path / "labels" / split
                images  = [f for f in img_dir.iterdir() if f.is_file()]
                logging.info(f"[{split}] checking {len(images)} images ...")

                for img_path in images:
                    label_path = lbl_dir / (img_path.stem + ".txt")

                    # no label file = negative sample = VALID
                    if not label_path.exists():
                        continue

                    # corrupt image
                    if cv2.imread(str(img_path)) is None:
                        logging.error(f"Corrupt image: {img_path.name}")
                        return False

                    # invalid label
                    if not self.validate_label_file(label_path):
                        return False

                logging.info(f"[{split}] passed")

            logging.info("Dataset validation passed")
            return True

        except Exception as e:
            raise DefectException(e, sys)

    # ================================================================
    # 3. GENERATE DATASET REPORT
    # ================================================================
    def generate_dataset_report(self) -> dict:
        try:
            stats = {
                "yolo_dataset_path"  : str(self.dataset_path),
                "num_classes"        : self.num_classes,
                "class_names"        : CLASS_NAMES,
                "train_images"       : 0,
                "val_images"         : 0,
                "train_positives"    : 0,
                "train_negatives"    : 0,
                "val_positives"      : 0,
                "val_negatives"      : 0,
                "total_bboxes"       : 0,
                "empty_label_files"  : 0,
                "corrupt_images"     : 0,
                "class_distribution" : {},
                "avg_bbox_width"     : 0.0,
                "avg_bbox_height"    : 0.0,
                "train_neg_pos_ratio": "inf",
                "val_neg_pos_ratio"  : "inf",
            }

            total_w = total_h = count_bbox = 0

            for split in ["train", "val"]:
                img_dir = self.dataset_path / "images" / split
                lbl_dir = self.dataset_path / "labels" / split

                for img_path in img_dir.iterdir():
                    if not img_path.is_file():
                        continue

                    if split == "train":
                        stats["train_images"] += 1
                    else:
                        stats["val_images"] += 1

                    if cv2.imread(str(img_path)) is None:
                        stats["corrupt_images"] += 1
                        continue

                    label_path = lbl_dir / (img_path.stem + ".txt")

                    if not label_path.exists():
                        stats[f"{split}_negatives"] += 1
                        continue

                    content = label_path.read_text().strip()

                    if not content:
                        stats["empty_label_files"] += 1
                        stats[f"{split}_negatives"] += 1
                        continue

                    stats[f"{split}_positives"] += 1

                    for line in content.splitlines():
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        class_id = int(parts[0])
                        w = float(parts[3])
                        h = float(parts[4])
                        stats["class_distribution"].setdefault(class_id, 0)
                        stats["class_distribution"][class_id] += 1
                        stats["total_bboxes"] += 1
                        total_w    += w
                        total_h    += h
                        count_bbox += 1

            if count_bbox > 0:
                stats["avg_bbox_width"]  = round(total_w / count_bbox, 6)
                stats["avg_bbox_height"] = round(total_h / count_bbox, 6)

            if stats["train_positives"] > 0:
                stats["train_neg_pos_ratio"] = round(
                    stats["train_negatives"] / stats["train_positives"], 2
                )
            if stats["val_positives"] > 0:
                stats["val_neg_pos_ratio"] = round(
                    stats["val_negatives"] / stats["val_positives"], 2
                )

            return stats

        except Exception as e:
            raise DefectException(e, sys)

    # ================================================================
    # PIPELINE ENTRY POINT
    # ================================================================
    def initiate_data_validation(self) -> DataValidationArtifact:
        logging.info("=" * 56)
        logging.info("  DataValidation — YOLO dataset check")
        logging.info("=" * 56)

        try:
            status = self.validate_yolo_dataset()
            stats  = self.generate_dataset_report()

            # save report
            with open(self.report_file, "w") as f:
                f.write("=" * 56 + "\n")
                f.write("  DefectVision — Data Validation Report\n")
                f.write("=" * 56 + "\n\n")
                for k, v in stats.items():
                    f.write(f"  {k:<25}: {v}\n")
                f.write(f"\n  validation_status     : {status}\n")

            # save status
            with open(self.status_file, "w") as f:
                f.write(f"validation_status: {status}\n")

            logging.info(
                f"\n  TRAIN  images: {stats['train_images']} "
                f"pos: {stats['train_positives']} "
                f"neg: {stats['train_negatives']} "
                f"ratio: {stats['train_neg_pos_ratio']}:1"
                f"\n  VAL    images: {stats['val_images']} "
                f"pos: {stats['val_positives']} "
                f"neg: {stats['val_negatives']}"
                f"\n  total bboxes : {stats['total_bboxes']}"
                f"\n  class dist   : {stats['class_distribution']}"
                f"\n  corrupt imgs : {stats['corrupt_images']}"
            )
            logging.info(f"  Report saved : {self.report_file}")
            logging.info(f"  Status       : {status}")
            logging.info("=" * 56)

            # return all 3 fields DataValidationArtifact requires
            return DataValidationArtifact(
                validation_status   = status,
                report_file_path    = self.report_file,
                data_validation_dir = self.data_validation_dir,
            )

        except Exception as e:
            raise DefectException(e, sys)
            anushka21854333@gmail.com