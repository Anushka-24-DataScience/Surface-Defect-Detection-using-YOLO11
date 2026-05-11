"""
DefectVision/pipeline/training_pipeline.py
==========================================
Orchestrates all pipeline stages in order.

Active stages:
    1. Data Ingestion      - download + extract KolektorSDD2
    2. Data Preprocessing  - denoise, CLAHE, sharpen, letterbox 640x640
    [Conversion]           - utils/convert.py  (separate utility, run manually)
    3. Data Validation     - validate YOLO dataset structure + labels
    4. Model Trainer       - train yolo11n / yolo11s / yolo11m, pick best
"""

import sys
import os
import glob

from DefectVision.logger    import logging
from DefectVision.exception import DefectException

from DefectVision.constant.training_pipeline import (
    CLASS_NAMES,
    ARTIFACTS_DIR,
)

from DefectVision.components.data_ingestion     import DataIngestion
from DefectVision.components.data_preprocessing import DataPreprocessing
from DefectVision.components.data_validation    import DataValidation
from DefectVision.components.model_trainer      import ModelTrainer   # ← NEW

from DefectVision.entity.config_entity import (
    DataIngestionConfig,
    DataPreprocessingConfig,
    DataValidationConfig,
    ModelTrainerConfig,                # ← NEW
    training_pipeline_config,
)

from DefectVision.entity.artifacts_entity import (
    DataIngestionArtifact,
    DataPreprocessingArtifact,
    DataValidationArtifact,
    DataConversionArtifact,
    ModelTrainerArtifact,              # ← NEW
)


class TrainPipeline:
    """
    Single TrainPipeline class.

    All configs are instantiated with no arguments — they derive
    every path from the shared training_pipeline_config singleton
    (defined at module level in config_entity.py) so all stages
    always use the same artifacts/<timestamp>/ folder.
    """

    def __init__(self):
        self.data_ingestion_config     = DataIngestionConfig()
        self.data_preprocessing_config = DataPreprocessingConfig()
        self.data_validation_config    = DataValidationConfig()
        self.model_trainer_config      = ModelTrainerConfig()   # ← NEW

        logging.info(
            f"Pipeline initialized — "
            f"artifacts dir: {training_pipeline_config.artifacts_dir}"
        )

    # ================================================================
    # STAGE 1 — DATA INGESTION
    # ================================================================
    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Starting Data Ingestion")
            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )
            artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Data Ingestion completed")
            return artifact
        except Exception as e:
            raise DefectException(e, sys)

    # ================================================================
    # STAGE 2 — DATA PREPROCESSING
    # ================================================================
    def start_data_preprocessing(self) -> DataPreprocessingArtifact:
        try:
            logging.info("Starting Data Preprocessing")
            preprocessing = DataPreprocessing(
                config=self.data_preprocessing_config
            )
            artifact = preprocessing.initiate_data_preprocessing()
            logging.info("Data Preprocessing completed")
            return artifact
        except Exception as e:
            raise DefectException(e, sys)

    # ================================================================
    # [UTILITY] GET YOLO DATASET PATH
    # Not a pipeline stage — conversion runs as utils/convert.py
     # ── [Utility] Conversion ─────────────────────────────────
            # Run manually: python DefectVision/utils/convert.py
            # Creates: artifacts/*/data_conversion/yolo_dataset/
    # ================================================================
    def get_yolo_dataset_path(self) -> str:
        """
        Finds the most recently created yolo_dataset under artifacts/.
        Works even when the dataset was created in a previous run
        with a different timestamp.
        """
        pattern    = os.path.join(
            ARTIFACTS_DIR, "*", "data_conversion", "yolo_dataset"
        )
        candidates = sorted(glob.glob(pattern))

        if not candidates:
            raise FileNotFoundError(
                "YOLO dataset not found under "
                "artifacts/*/data_conversion/yolo_dataset\n"
                "Run utils/convert.py first."
            )

        yolo_path = candidates[-1]
        logging.info(f"Using YOLO dataset: {yolo_path}")
        return yolo_path

    # ================================================================
    # STAGE 3 — DATA VALIDATION
    # ================================================================
    def start_data_validation(
        self, yolo_dataset_path: str
    ) -> DataValidationArtifact:
        try:
            logging.info("Starting Data Validation")

            data_conversion_artifact = DataConversionArtifact(
                yolo_dataset_path=yolo_dataset_path
            )

            validation = DataValidation(
                data_conversion_artifact = data_conversion_artifact,
                data_validation_dir      = self.data_validation_config.data_validation_dir,
                num_classes              = len(CLASS_NAMES),
            )

            artifact = validation.initiate_data_validation()
            logging.info("Data Validation completed")
            return artifact

        except Exception as e:
            raise DefectException(e, sys)

    # ================================================================
    # STAGE 4 — MODEL TRAINER                                  ← NEW
    # ================================================================
    def start_model_trainer(
        self, yolo_dataset_path: str
    ) -> ModelTrainerArtifact:
        """
        Trains yolo11n, yolo11s, yolo11m with identical hyperparameters.
        Compares all 3 by mAP50 and saves the winner as best.pt.

        Each variant saves last.pt after every epoch so training is
        always resumable if interrupted (GPU timeout, power cut, etc).

        Parameters
        ----------
        yolo_dataset_path : str
            Path to yolo_dataset/ folder containing data.yaml.
            Comes from get_yolo_dataset_path() or DataConversionArtifact.
        """
        try:
            logging.info("Starting Model Trainer")

            data_yaml_path = os.path.join(yolo_dataset_path, "data.yaml")

            if not os.path.exists(data_yaml_path):
                raise FileNotFoundError(
                    f"data.yaml not found at: {data_yaml_path}\n"
                    "Run utils/convert.py first."
                )

            trainer = ModelTrainer(
                config         = self.model_trainer_config,
                data_yaml_path = data_yaml_path,
            )
            artifact = trainer.initiate_model_trainer()

            logging.info(
                f"Model Trainer completed — "
                f"best: {artifact.best_model_name}  "
                f"mAP50: {artifact.map50}"
            )
            return artifact

        except Exception as e:
            raise DefectException(e, sys)

    # ================================================================
    # RUN PIPELINE
    # ================================================================
    def run_pipeline(self):
        try:
            logging.info("=" * 56)
            logging.info("  TrainPipeline starting")
            logging.info("=" * 56)

            # ── Stage 3: Data Validation ─────────────────────────────
            yolo_dataset_path = self.get_yolo_dataset_path()

            validation_artifact = self.start_data_validation(
                yolo_dataset_path=yolo_dataset_path
            )

            if not validation_artifact.validation_status:
                raise Exception(
                    "Data Validation Failed — check logs and "
                    f"{self.data_validation_config.report_file_path}"
                )

            # ── Stage 4: Model Trainer ───────────────────────────────
            model_trainer_artifact = self.start_model_trainer(
                yolo_dataset_path=yolo_dataset_path
            )

            logging.info("=" * 56)
            logging.info("  Pipeline completed successfully")
            logging.info("=" * 56)
            logging.info(f"  Best model     : {model_trainer_artifact.best_model_name}")
            logging.info(f"  best.pt        : {model_trainer_artifact.trained_model_path}")
            logging.info(f"  mAP50          : {model_trainer_artifact.map50}")
            logging.info(f"  mAP50-95       : {model_trainer_artifact.map50_95}")
            logging.info(f"  Precision      : {model_trainer_artifact.precision}")
            logging.info(f"  Recall         : {model_trainer_artifact.recall}")
            logging.info(f"  Comparison CSV : {model_trainer_artifact.model_comparison_csv}")

        except Exception as e:
            raise DefectException(e, sys)


if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()