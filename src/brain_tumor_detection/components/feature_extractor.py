import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

from brain_tumor_detection.entity.config_entity import (
    FeatureExtractionConfig
)

from brain_tumor_detection.utils.common import (
    create_directory
)

from brain_tumor_detection.logger import logger
from brain_tumor_detection.exception import CustomException


class FeatureExtractor:

    def __init__(self,config: FeatureExtractionConfig):

        self.config = config

        logger.info(
            "Initializing FeatureExtractor."
        )

        create_directory(
            self.config.root_dir
        )

        self.model_path = Path(
            "artifacts/model_trainer/resnet50_finetuned.keras"
        )

        self.model = self._load_model()

        self.feature_model = self._build_feature_model()

    def _load_model(self):

        try:

            logger.info(
                f"Loading trained model from "
                f"{self.model_path}"
            )

            if not self.model_path.exists():

                raise FileNotFoundError(
                    f"Model not found at "
                    f"{self.model_path}"
                )

            model = load_model(
                self.model_path,
                custom_objects={
                    "preprocess_input": preprocess_input
                },
                compile=False
            )

            logger.info(
                "Trained model loaded successfully."
            )

            return model

        except Exception as e:

            logger.exception(
                "Failed to load trained model."
            )

            raise CustomException(
                e,
                sys
            )

    def _build_feature_model(self):

        try:

            logger.info(
                "Building feature extraction model."
            )

            
            dummy_input = tf.zeros(
                shape=(1, 224, 224, 3),
                dtype=tf.float32
            )

            self.model(
                dummy_input,
                training=False
            )

            logger.info(
                "Trained model initialized successfully."
            )

            feature_layer = self.model.get_layer(
                "global_average_pooling2d"
            )

            feature_model = tf.keras.Model(
                inputs=self.model.inputs,
                outputs=feature_layer.output
            )

            logger.info(
                "Feature extraction model created successfully."
            )

            logger.info(
                f"Feature model output shape: "
                f"{feature_model.output_shape}"
            )

            return feature_model

        except Exception as e:

            logger.exception(
                "Failed to build feature extraction model."
            )

            raise CustomException(e,sys)

    def _preprocess_image(self,image_path: Path):

        try:

            logger.info(
                f"Preprocessing image: {image_path}"
            )

            if not image_path.exists():

                raise FileNotFoundError(
                    f"Image not found at "
                    f"{image_path}"
                )

            image = tf.keras.utils.load_img(
                image_path,
                target_size=(224, 224),
                color_mode="rgb"
            )

            image_array = (
                tf.keras.utils.img_to_array(
                    image
                )
            )

            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            logger.info(
                "Image loading and resizing completed. "
                "Model will perform preprocessing internally."
            )

            return image_array

        except Exception as e:

            logger.exception(
                "Image preprocessing failed."
            )

            raise CustomException(e,sys)

    def extract_feature(self,image_path: Path) -> np.ndarray:

        try:

            image_path = Path(
                image_path
            )

            logger.info(
                f"Starting feature extraction for: "
                f"{image_path}"
            )

            processed_image = (
                self._preprocess_image(
                    image_path
                )
            )

            embedding = (
                self.feature_model.predict(
                    processed_image,
                    verbose=0
                )[0]
            )

            embedding = np.asarray(
                embedding,
                dtype=np.float32
            )

            logger.info(
                "Feature extraction completed. "
                f"Embedding shape: {embedding.shape}"
            )

            return embedding

        except Exception as e:

            logger.exception(
                "Feature extraction failed."
            )

            raise CustomException(e,sys)

    def initiate_feature_extraction(self,image_path: Path) -> np.ndarray:

        return self.extract_feature(
            image_path)