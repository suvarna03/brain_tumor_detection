import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

from brain_tumor_detection.entity.config_entity import (
    ModelInferenceConfig
)

from brain_tumor_detection.logger import logger
from brain_tumor_detection.exception import CustomException


class ModelInference:

    def __init__(
        self,
        config: ModelInferenceConfig
    ):

        self.config = config

        logger.info(
            "Initializing ModelInference."
        )

        self.model = self._load_model()

    def _load_model(self):

        try:

            logger.info(
                f"Loading trained model from "
                f"{self.config.model_path}"
            )

            if not self.config.model_path.exists():

                raise FileNotFoundError(
                    f"Model not found at "
                    f"{self.config.model_path}"
                )

            model = load_model(
                self.config.model_path,
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
                target_size=self.config.image_size,
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

      

    def predict(self,image_path: Path) -> dict:

        try:

            image_path = Path(
                image_path
            )

            logger.info(
                f"Starting inference for: "
                f"{image_path}"
            )

            processed_image = (
                self._preprocess_image(
                    image_path
                )
            )

            probabilities = (
                self.model.predict(
                    processed_image,
                    verbose=0
                )[0]
            )

            predicted_index = int(
                np.argmax(
                    probabilities
                )
            )

            predicted_class = (
                self.config.class_names[
                    predicted_index
                ]
            )

            confidence = float(
                probabilities[
                    predicted_index
                ]
            )

            class_probabilities = {
                class_name: float(
                    probability
                )
                for class_name, probability
                in zip(
                    self.config.class_names,
                    probabilities
                )
            }

            result = {

                "predicted_class":
                    predicted_class,

                "confidence":
                    confidence,

                "probabilities":
                    class_probabilities
            }

            logger.info(
                f"Inference completed. "
                f"Prediction: {predicted_class}, "
                f"Confidence: {confidence:.4f}"
            )

            return result

        except Exception as e:

            logger.exception(
                "Model inference failed."
            )

            raise CustomException(
                e,
                sys
            )

    def initiate_model_inference(
        self,
        image_path: Path
    ) -> dict:

        return self.predict(
            image_path
        )