"""
DefectVision/pipeline/training_pipeline.py
==========================================
Orchestrates all pipeline stages in order.

Current active stages:
    1. Data Ingestion      - download + extract KolektorSDD2
    2. Data Preprocessing  - denoise, CLAHE, sharpen, letterbox 640x640
    [Conversion]           - utils/convert.py  (separate utility, run manually)
    3. Data Validation     - validate YOLO dataset structure + labels

Why conversion is a utility not a pipeline stage:
    Conversion depends on the target model format (YOLO needs .txt labels,
    COCO needs JSON, etc). Keeping it as utils/convert.py makes the pipeline
    model-agnostic and reusable. The pipeline validates whatever YOLO
    dataset exists on disk without caring how it was created.
"""

import sys
import os
import glob

from DefectVision.logger import logging
from DefectVision.exception import DefectException

from DefectVision.constant.training_pipeline import (
    CLASS_NAMES,
    ARTIFACTS_DIR,
)

from DefectVision.components.data_ingestion    import DataIngestion
from DefectVision.components.data_preprocessing import DataPreprocessing
from DefectVision.components.data_validation   import DataValidation

from DefectVision.entity.config_entity import (
    DataIngestionConfig,
    DataPreprocessingConfig,
    DataValidationConfig,
    training_pipeline_config,
)

from DefectVision.entity.artifacts_entity import (
    DataIngestionArtifact,
    DataPreprocessingArtifact,
    DataValidationArtifact,
    DataConversionArtifact,
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
        # No arguments needed — paths come from training_pipeline_config
        self.data_ingestion_config     = DataIngestionConfig()
        self.data_preprocessing_config = DataPreprocessingConfig()
        self.data_validation_config    = DataValidationConfig()

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
    # This method simply locates the most recent converted dataset
    # ================================================================
    def get_yolo_dataset_path(self) -> str:
        """
        Finds the most recently created yolo_dataset under artifacts/.
        Searches all timestamp folders — works even when the dataset
        was created in a previous run with a different timestamp.
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
        """
        Validates the YOLO dataset produced by utils/convert.py.

        DataValidation.__init__ expects:
            data_conversion_artifact : DataConversionArtifact
            data_validation_dir      : str   (path to save report + status)
            num_classes              : int   (1 for KolektorSDD2)
        """
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
    # RUN PIPELINE
    # ================================================================
    def run_pipeline(self):
        try:
            logging.info("=" * 56)
            logging.info("  TrainPipeline starting")
            logging.info("=" * 56)

            # ── Stage 1: Data Ingestion ──────────────────────────────
            # Uncomment to re-download dataset
            # ingestion_artifact = self.start_data_ingestion()

            # ── Stage 2: Data Preprocessing ─────────────────────────
            # Uncomment to re-preprocess images
            # preprocessing_artifact = self.start_data_preprocessing()

            # ── [Utility] Conversion ─────────────────────────────────
            # Run manually: python DefectVision/utils/convert.py
            # Creates: artifacts/*/data_conversion/yolo_dataset/

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

            logging.info("Pipeline completed successfully")
            logging.info(
                f"Validation report: "
                f"{self.data_validation_config.report_file_path}"
            )

        except Exception as e:
            raise DefectException(e, sys)


if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()