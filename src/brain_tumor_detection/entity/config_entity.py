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

@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    trained_model_path: Path


@dataclass(frozen=True)
class FeatureExtractionConfig:
    root_dir: Path
    feature_file: Path


@dataclass(frozen=True)
class FAISSConfig:
    root_dir: Path
    index_file: Path