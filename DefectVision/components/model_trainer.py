"""
DefectVision/components/model_trainer.py
=========================================
Stage 4 — trains yolo11n, yolo11s, yolo11m and picks the best.

Interview explanation:
    "We train three YOLO11 variants with identical hyperparameters so the
     comparison is fair. Each variant saves last.pt after every epoch so
     training is always resumable after interruption. After all three finish,
     we compare by mAP50 and promote the winner to best.pt."

Run directly (standalone):
    python -m DefectVision.components.model_trainer
"""

import os
import sys
import glob
import shutil
import time
from pathlib import Path

import pandas as pd

from DefectVision.logger    import logging
from DefectVision.exception import DefectException

from DefectVision.entity.config_entity    import ModelTrainerConfig
from DefectVision.entity.artifacts_entity import ModelTrainerArtifact
from DefectVision.constant.training_pipeline import (
    ARTIFACTS_DIR,
)

DATA_CONVERSION_DIR_NAME = "data_conversion"
YOLO_DATASET_DIR_NAME    = "yolo_dataset"

# Defect-specific augmentation overrides
# KolektorSDD2 is grayscale — disable colour jitter, keep geometry augments
TRAIN_KWARGS = dict(
    hsv_h   = 0.0,   # no hue shift   (grayscale)
    hsv_s   = 0.0,   # no saturation  (grayscale)
    hsv_v   = 0.3,   # slight brightness variation OK
    fliplr  = 0.5,
    flipud  = 0.3,
    degrees = 10.0,
    scale   = 0.3,
)


class ModelTrainer:
    """
    Trains all YOLO11 variants and returns the best one.

    Parameters
    ----------
    config         : ModelTrainerConfig
    data_yaml_path : str   path to data.yaml from yolo_dataset/
    """

    def __init__(self, config: ModelTrainerConfig, data_yaml_path: str):
        self.config         = config
        self.data_yaml_path = data_yaml_path

    # ── public entry point ────────────────────────────────────────────
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics not installed. Run: pip install ultralytics"
            )

        try:
            if not os.path.exists(self.data_yaml_path):
                raise FileNotFoundError(
                    f"data.yaml not found: {self.data_yaml_path}\n"
                    "Run utils/convert.py first."
                )

            os.makedirs(self.config.model_trainer_dir, exist_ok=True)

            results_summary = self._load_existing_results()
            finished_models = {r["model"] for r in results_summary}

            logging.info(
                f"Models already finished: {finished_models if finished_models else 'none'}"
            )

            # ── train each variant ────────────────────────────────────
            for weight_file in self.config.pretrained_weights:
                model_name  = Path(weight_file).stem   # yolo11n / yolo11s / yolo11m
                variant_dir = Path(self.config.model_trainer_dir) / model_name
                variant_dir.mkdir(parents=True, exist_ok=True)

                logging.info(f"{'='*50}")
                logging.info(f"Variant: {model_name}")
                logging.info(f"{'='*50}")

                # Skip if already fully trained
                if model_name in finished_models:
                    logging.info(f"SKIP {model_name} — already complete")
                    self._restore_best_locally(model_name, variant_dir)
                    continue

                # Resume or start fresh
                start      = time.time()
                local_last = variant_dir / "last.pt"
                result     = self._train_variant(
                    YOLO, weight_file, model_name, local_last, variant_dir
                )
                elapsed = round((time.time() - start) / 60, 1)

                # Save checkpoints to variant_dir
                self._copy_checkpoint(
                    Path(self.config.model_trainer_dir) / model_name / "weights" / "best.pt",
                    variant_dir / "best.pt",
                )
                self._copy_checkpoint(
                    Path(self.config.model_trainer_dir) / model_name / "weights" / "last.pt",
                    variant_dir / "last.pt",
                )

                # Collect and save metrics immediately
                row = self._extract_metrics(result, model_name, elapsed, variant_dir)
                results_summary.append(row)
                self._save_csv(results_summary)

                logging.info(
                    f"{model_name} complete — "
                    f"mAP50={row['mAP50']}  "
                    f"precision={row['precision']}  "
                    f"recall={row['recall']}  "
                    f"time={elapsed}min"
                )

            if not results_summary:
                raise RuntimeError("No models trained successfully.")

            # ── pick winner ───────────────────────────────────────────
            best_row  = max(results_summary, key=lambda x: x["mAP50"])
            best_name = best_row["model"]
            best_src  = Path(best_row["best_pt"])
            best_dst  = Path(self.config.model_trainer_dir) / "best.pt"

            shutil.copy2(str(best_src), str(best_dst))
            csv_path = self._save_csv(results_summary)

            # Log comparison table
            df = pd.DataFrame(results_summary)
            logging.info("\n" + "="*50)
            logging.info("MODEL COMPARISON")
            logging.info("="*50)
            logging.info(
                "\n" + df[["model","mAP50","mAP50_95","precision","recall","train_mins"]].to_string(index=False)
            )
            logging.info(f"Best model : {best_name}  (mAP50={best_row['mAP50']})")
            logging.info(f"best.pt    : {best_dst}")

            artifact = ModelTrainerArtifact(
                trained_model_path   = str(best_dst),
                best_model_name      = best_name,
                model_comparison_csv = csv_path,
                map50                = best_row["mAP50"],
                map50_95             = best_row["mAP50_95"],
                precision            = best_row["precision"],
                recall               = best_row["recall"],
            )

            logging.info("Exited initiate_model_trainer method of ModelTrainer class")
            logging.info(f"Model trainer artifact: {artifact}")
            return artifact

        except Exception as e:
            raise DefectException(e, sys)

    # ── private helpers ───────────────────────────────────────────────

    def _train_variant(self, YOLO, weight_file, model_name, local_last, variant_dir):
        """Resume from last.pt if it exists, else start fresh."""
        if local_last.exists():
            logging.info(f"RESUME {model_name} from: {local_last}")
            model  = YOLO(str(local_last))
            result = model.train(resume=True)
        else:
            logging.info(f"START  {model_name} from scratch")
            model  = YOLO(weight_file)
            result = model.train(
                data     = self.data_yaml_path,
                epochs   = self.config.no_epochs,
                imgsz    = self.config.img_size,
                batch    = self.config.batch_size,
                patience = self.config.patience,
                workers  = self.config.workers,
                name     = model_name,
                project  = self.config.model_trainer_dir,
                exist_ok = True,
                **TRAIN_KWARGS,
            )
        return result

    def _extract_metrics(self, result, model_name, elapsed, variant_dir) -> dict:
        m = result.results_dict
        return {
            "model"      : model_name,
            "mAP50"      : round(m.get("metrics/mAP50(B)",    0.0), 4),
            "mAP50_95"   : round(m.get("metrics/mAP50-95(B)", 0.0), 4),
            "precision"  : round(m.get("metrics/precision(B)",0.0), 4),
            "recall"     : round(m.get("metrics/recall(B)",   0.0), 4),
            "train_mins" : elapsed,
            "best_pt"    : str(variant_dir / "best.pt"),
        }

    def _copy_checkpoint(self, src: Path, dst: Path):
        if src.exists():
            shutil.copy2(str(src), str(dst))
            mb = round(dst.stat().st_size / 1024 / 1024, 1)
            logging.info(f"Saved {dst.name} ({mb} MB) -> {dst.parent}")

    def _save_csv(self, results_summary: list) -> str:
        csv_path = os.path.join(
            self.config.model_trainer_dir,
            self.config.results_csv,
        )
        pd.DataFrame(results_summary).to_csv(csv_path, index=False)
        logging.info(f"Results CSV saved: {csv_path}")
        return csv_path

    def _load_existing_results(self) -> list:
        """Load previously saved results so finished models are not retrained."""
        csv_path = os.path.join(
            self.config.model_trainer_dir,
            self.config.results_csv,
        )
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            logging.info(f"Loaded {len(df)} existing results from {csv_path}")
            return df.to_dict("records")
        return []

    def _restore_best_locally(self, model_name: str, variant_dir: Path):
        """
        If best.pt is missing locally (e.g. after a machine restart),
        it should already be in variant_dir from the previous run.
        Log a warning if it's gone.
        """
        best = variant_dir / "best.pt"
        if not best.exists():
            logging.warning(
                f"best.pt missing for {model_name} at {best}. "
                "It may have been deleted. Re-run to retrain this variant."
            )


# =====================================================================
# STANDALONE ENTRY POINT
# python -m DefectVision.components.model_trainer
# =====================================================================

def main():
    pattern    = os.path.join(
        ARTIFACTS_DIR, "*",
        DATA_CONVERSION_DIR_NAME,
        YOLO_DATASET_DIR_NAME,
        "data.yaml",
    )
    candidates = sorted(glob.glob(pattern))

    if not candidates:
        logging.error(
            "No data.yaml found under artifacts/. "
            "Run utils/convert.py first."
        )
        sys.exit(1)

    data_yaml_path = candidates[-1]
    logging.info(f"Using data.yaml: {data_yaml_path}")

    config  = ModelTrainerConfig()
    trainer = ModelTrainer(config=config, data_yaml_path=data_yaml_path)
    trainer.initiate_model_trainer()


if __name__ == "__main__":
    main()