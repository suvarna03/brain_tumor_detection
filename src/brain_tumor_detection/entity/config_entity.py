from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    dataset_name: str
    dataset_dir: Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    status_file: Path
    dataset_dir: Path

    expected_classes: list
    expected_num_classes: int

    allowed_formats: list
    allowed_modes: list

    pixel_min: int
    pixel_max: int

@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    trained_model_path: Path

    image_size: tuple
    batch_size: int
    epochs: int
    learning_rate: float

    model_name: str
    include_top: bool
    weights: str
    classes: int
    color_mode: str
    normalize: bool


@dataclass(frozen=True)
class FeatureExtractionConfig:
    root_dir: Path
    feature_file: Path


@dataclass(frozen=True)
class FAISSConfig:
    root_dir: Path
    index_file: Path