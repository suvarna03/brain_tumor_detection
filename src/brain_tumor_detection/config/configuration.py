from pathlib import Path

from brain_tumor_detection.constants import (
    CONFIG_FILE_PATH,
    PARAMS_FILE_PATH,
    SCHEMA_FILE_PATH
)

from brain_tumor_detection.utils.common import (
    read_yaml,
    create_directory
)

from brain_tumor_detection.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    ModelTrainingConfig,
    FeatureExtractionConfig,
    FAISSConfig
)


class ConfigurationManager:

    def __init__(self):

        self.config = read_yaml(CONFIG_FILE_PATH)
        self.params = read_yaml(PARAMS_FILE_PATH)
        self.schema = read_yaml(SCHEMA_FILE_PATH)

        create_directory(
            Path(self.config.artifacts_root)
        )

    def get_data_ingestion_config(self) -> DataIngestionConfig:

        data_ingestion = self.config.data_ingestion

        data_ingestion_config =DataIngestionConfig(
            root_dir=Path(data_ingestion.root_dir),
            dataset_name=data_ingestion.dataset_name,
            dataset_dir=Path(data_ingestion.dataset_dir)
                )

        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:

        data_validation = self.config.data_validation

        data_validation_config = DataValidationConfig(
            root_dir=Path(data_validation.root_dir),
            status_file=Path(data_validation.status_file),
            dataset_dir=Path(data_validation.dataset_dir),

            expected_classes=self.schema.DATA_VALIDATION.CLASSIFICATION_CLASSES,
            expected_num_classes=self.schema.expected_num_classes,

            allowed_formats=self.schema.allowed_formats,
            allowed_modes=self.schema.allowed_modes,

            pixel_min=self.schema.pixel_range.min,
            pixel_max=self.schema.pixel_range.max
            )

        return data_validation_config

    def get_model_training_config(self) -> ModelTrainingConfig:

        model_training = self.config.model_training
        params = self.params

        model_training_config = ModelTrainingConfig(

            root_dir=Path(model_training.root_dir),
            trained_model_path=Path(model_training.trained_model_path),
            image_size=tuple(params.IMAGE_SIZE),
            batch_size=params.BATCH_SIZE,
            epochs=params.EPOCHS,
            learning_rate=params.LEARNING_RATE,
            model_name=params.MODEL_NAME,
            include_top=params.INCLUDE_TOP,
            weights=params.WEIGHTS,
            classes=params.CLASSES,
            color_mode=params.COLOR_MODE,
            normalize=params.NORMALIZE
          )

        return model_training_config
    
    def get_feature_extraction_config(self) -> FeatureExtractionConfig:

        feature_extraction = self.config.feature_extraction
        feature_extraction_config = FeatureExtractionConfig(
            root_dir = Path(feature_extraction.root_dir),
            feature_file = Path(feature_extraction.feature_file)
        )

        return feature_extraction_config
    
    def get_faiss_config(self) -> FAISSConfig:
        
        faiss_section = self.config.faiss_index
        faiss_index_config = FAISSConfig(
            root_dir = Path(faiss_section.root_dir),
            index_file = Path(faiss_section.index_file)
        )
        return faiss_index_config