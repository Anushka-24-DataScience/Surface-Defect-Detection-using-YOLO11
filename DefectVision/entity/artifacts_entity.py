# from dataclasses import dataclass

# @dataclass
# class DataIngestionArtifact:
#     data_zip_file_path:str
#     feature_store_path:str


# # @dataclass
# # class DataPreprocessingArtifact:
# #     """
# #     Produced by DataPreprocessing component.
 
# #     preprocessed_dir  : root folder containing images/ and labels/ subfolders
# #                         ready for YOLO11 training.
# #     train_img_dir     : preprocessed_dir/images/train
# #     val_img_dir       : preprocessed_dir/images/val
# #     train_label_dir   : preprocessed_dir/labels/train  (letterbox-adjusted)
# #     val_label_dir     : preprocessed_dir/labels/val    (letterbox-adjusted)
# #     original_img_size : (width, height) of raw images BEFORE letterbox —
# #                         stored so label_adjustment can recompute padding offset
# #     target_size       : (width, height) images were padded to (always 640x640)
# #     """
# #     preprocessed_dir:  str
# #     train_img_dir:     str
# #     val_img_dir:       str
# #     train_label_dir:   str
# #     val_label_dir:     str
# #     original_img_size: tuple   # (w, h)  e.g. (230, 630)
# #     target_size:       tuple   # (w, h)  e.g. (640, 640)
 

# @dataclass
# class DataPreprocessingArtifact:
#     """
#     Produced by DataPreprocessing component.

#     Only images are produced here.
#     Labels are created AFTER this by utils/convert.py,
#     which reads preprocessed images + raw masks from feature_store.

#     preprocessed_dir : root folder  e.g. artifacts/<ts>/data_preprocessing/processed
#     train_img_dir    : preprocessed_dir/images/train   (640x640 BGR images)
#     val_img_dir      : preprocessed_dir/images/val     (640x640 BGR images)
#     target_size      : (width, height) images were resized to — always (640, 640)
#     """
#     preprocessed_dir : str
#     train_img_dir    : str
#     val_img_dir      : str
#     target_size      : tuple    # (640, 640)

# @dataclass
# class DataConversionArtifact:
#     yolo_dataset_dir : str    # full path to yolo_dataset/
#     data_yaml_path   : str    # full path to data.yaml
#     train_img_dir    : str
#     val_img_dir      : str
#     train_label_dir  : str
#     val_label_dir    : str
#     total_train      : int
#     total_val        : int


# @dataclass
# class DataConversionArtifact:
#     yolo_dataset_path: str


# @dataclass
# class DataValidationArtifact:
#     validation_status: bool
#     report_file_path: str
#     data_validation_dir:str

from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    data_zip_file_path: str
    feature_store_path: str


@dataclass
class DataPreprocessingArtifact:
    """
    Produced by DataPreprocessing component.
    Images only — labels are created later by utils/convert.py.
    """
    preprocessed_dir : str
    train_img_dir    : str
    val_img_dir      : str
    target_size      : tuple    # (640, 640)


@dataclass
class DataConversionArtifact:
    """
    Produced by utils/convert.py (utility, not a pipeline stage).
    Only yolo_dataset_path is needed — pipeline uses it to locate the dataset.
    """
    yolo_dataset_path: str


@dataclass
class DataValidationArtifact:
    """
    Produced by DataValidation component.
    Must match exactly what initiate_data_validation() returns.
    """
    validation_status   : bool
    report_file_path    : str
    data_validation_dir : str




# ── artifacts_entity.py ────────────────────────────────────────────────────────
@dataclass
class ModelTrainerArtifact:
    """
    Returned by ModelTrainer.initiate_model_trainer().
    Contains path and metrics of the best model across all variants.
    """
    trained_model_path  : str    # path to best.pt  (winner)
    best_model_name     : str    # e.g. "yolo11s"
    model_comparison_csv: str    # path to model_comparison.csv
    map50               : float  # mAP@0.5 of best model
    map50_95            : float  # mAP@0.5:0.95 of best model
    precision           : float
    recall              : float