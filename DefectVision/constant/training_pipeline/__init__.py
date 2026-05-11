import os

ARTIFACTS_DIR: str = "artifacts"


"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_DIR_NAME: str = "data_ingestion"

DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"

DATA_DOWNLOAD_URL: str = "https://huggingface.co/buckets/AnushkaSrivastava/defect/resolve/KolektorSDD2.zip?download=true"

# ─── MASK NAMING ──────────────────────────────────────────────────────────────
# image: 10000   mask: 10000_GT  (any extension)
MASK_SUFFIX    = "_GT"

# ─── CLASS ────────────────────────────────────────────────────────────────────
CLASS_NAMES    = ["defect"]
CLASS_ID       = 0

# ─── CONVERSION ───────────────────────────────────────────────────────────────
MIN_CONTOUR_AREA           = 10    # px² — smaller blobs treated as mask noise
NEG_TO_POS_RATIO           = 5    # keep at most 5× negatives vs positives
AUGMENT_COPIES_PER_POSITIVE = 3   # extra augmented copies per positive (0=off)
RANDOM_SEED                = 42

# # ─── TRAINING ─────────────────────────────────────────────────────────────────
# YOLO_MODEL     = "yolo11n.pt"     # n=nano, s=small, m=medium, l=large, x=xlarge
# EPOCHS         = 100
# IMG_SIZE       = 640
# BATCH_SIZE     = 16
# PATIENCE       = 20               # early stopping patience
# WORKERS        = 4
# DEVICE         = ""               # "" = auto (GPU if available, else CPU)


"""
Data Preprocessing related constant
"""
DATA_PREPROCESSING_DIR_NAME: str = "data_preprocessing"
DATA_PREPROCESSING_PROCESSED_DIR: str = "processed"

PREPROCESSING_TARGET_SIZE: tuple = (640, 640)   # letterbox target (w, h)
PREPROCESSING_PAD_VALUE:   int   = 114          # YOLO standard gray pad


"""
Data Validation realted contant start with DATA_VALIDATION VAR NAME
"""


DATA_VALIDATION_DIR_NAME: str = "data_validation"

DATA_VALIDATION_STATUS_FILE = "status.txt"

DATA_VALIDATION_REPORT_FILE = "report.txt"

DATA_VALIDATION_REQUIRED_DIRS = [
    "images/train",
    "images/val",
    "labels/train",
    "labels/val",
]

DATA_VALIDATION_SAMPLE_DIR = "samples"
DATA_VALIDATION_OUTLIER_FILE = "outliers.txt"
YOLO_DATASET_PATH="C:\\Users\\admin\\Desktop\\Surface-Defect-Detection-using-YOLO11\\artifacts\\05_01_2026_22_50_34\\data_conversion\\yolo_dataset"
DATA_VALIDATION_DIR = "C:\\Users\\admin\\Desktop\\Surface-Defect-Detection-using-YOLO11\\artifacts\\05_01_2026_22_50_34\\data_validation"