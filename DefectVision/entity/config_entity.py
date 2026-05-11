import os
from dataclasses import dataclass, field
from datetime import datetime
from DefectVision.constant.training_pipeline import *

# ─────────────────────────────────────────────────────────────────────────────
# TIMESTAMP — created ONCE at module import time
# All configs below use this same instance so every stage shares
# the same artifacts/<timestamp>/ folder in one pipeline run
# ─────────────────────────────────────────────────────────────────────────────
TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


@dataclass
class TrainingPipelineConfig:
    artifacts_dir: str = os.path.join(ARTIFACTS_DIR, TIMESTAMP)


# Module-level singleton — import this in pipeline to get shared timestamp
training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()


# ─────────────────────────────────────────────────────────────────────────────
# DATA INGESTION CONFIG
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DataIngestionConfig:
    """
    All paths derived from training_pipeline_config — no args needed.
    Call as: DataIngestionConfig()
    """
    data_ingestion_dir: str = os.path.join(
        training_pipeline_config.artifacts_dir,
        DATA_INGESTION_DIR_NAME,
    )

    feature_store_file_path: str = os.path.join(
        training_pipeline_config.artifacts_dir,
        DATA_INGESTION_DIR_NAME,
        DATA_INGESTION_FEATURE_STORE_DIR,
    )

    data_download_url: str = DATA_DOWNLOAD_URL


# ─────────────────────────────────────────────────────────────────────────────
# DATA PREPROCESSING CONFIG
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DataPreprocessingConfig:
    """
    Input  : feature_store raw images (from DataIngestion)
    Output : preprocessed/images/train and /val  (640x640 BGR)

    No label paths — labels are created by utils/convert.py after this.
    Call as: DataPreprocessingConfig()
    """
    feature_store_path: str = os.path.join(
        training_pipeline_config.artifacts_dir,
        DATA_INGESTION_DIR_NAME,
        DATA_INGESTION_FEATURE_STORE_DIR,
    )

    preprocessed_dir: str = os.path.join(
        training_pipeline_config.artifacts_dir,
        DATA_PREPROCESSING_DIR_NAME,
        DATA_PREPROCESSING_PROCESSED_DIR,
    )

    target_size: tuple = PREPROCESSING_TARGET_SIZE   # (640, 640)
    pad_value:   int   = PREPROCESSING_PAD_VALUE     # 114


# ─────────────────────────────────────────────────────────────────────────────
# DATA VALIDATION CONFIG
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DataValidationConfig:
    """
    All paths derived from training_pipeline_config — no args needed.
    Call as: DataValidationConfig()
    """
    data_validation_dir: str = os.path.join(
        training_pipeline_config.artifacts_dir,
        DATA_VALIDATION_DIR_NAME,
    )

    status_file_path: str = os.path.join(
        training_pipeline_config.artifacts_dir,
        DATA_VALIDATION_DIR_NAME,
        "status.txt",
    )

    report_file_path: str = os.path.join(
        training_pipeline_config.artifacts_dir,
        DATA_VALIDATION_DIR_NAME,
        "report.txt",
    )



# ── TRAINING  CONFIG───────────────────────────────────────────────────────────
 
@dataclass
class ModelTrainerConfig:
    """
    Config for Stage 4 — Model Training.
 
    Reads  : artifacts/<ts>/data_conversion/yolo_dataset/data.yaml
    Writes : artifacts/<ts>/model_trainer/
                best.pt              ← best model across all 3 variants
                model_comparison.csv ← metrics table for all 3
                yolo11n/best.pt      ← individual weights
                yolo11n/last.pt      ← resume checkpoint
                yolo11s/best.pt
                yolo11s/last.pt
                yolo11m/best.pt
                yolo11m/last.pt
    """
    model_trainer_dir: str = os.path.join(
        training_pipeline_config.artifacts_dir,
        MODEL_TRAINER_DIR_NAME,
    )
 
    pretrained_weights: list = field(
        default_factory=lambda: MODEL_TRAINER_PRETRAINED_WEIGHTS
    )
 
    no_epochs   : int = MODEL_TRAINER_NO_EPOCHS
    batch_size  : int = MODEL_TRAINER_BATCH_SIZE
    img_size    : int = MODEL_TRAINER_IMG_SIZE
    patience    : int = MODEL_TRAINER_PATIENCE
    workers     : int = MODEL_TRAINER_WORKERS
    results_csv : str = MODEL_TRAINER_RESULTS_CSV