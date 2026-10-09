from pathlib import Path
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model

from brain_tumor_detection.entity.config_entity import ModelInferenceConfig
from brain_tumor_detection.exception import CustomException
from brain_tumor_detection.logger import logger


class ModelInference:

    def __init__(self, config: ModelInferenceConfig):
        self.config = config

        logger.info("Initializing ModelInference.")

        self.model = self._load_model()

    def _load_model(self):

        try:
            logger.info(
                f"Loading trained model from {self.config.model_path}"
            )

            if not self.config.model_path.exists():
                raise FileNotFoundError(
                    f"Model not found at {self.config.model_path}"
                )

            model = load_model(
                self.config.model_path,
                custom_objects={
                    "preprocess_input": preprocess_input
                },
                compile=False
            )

            logger.info("Trained model loaded successfully.")

            return model

        except Exception as e:
            logger.exception("Failed to load trained model.")
            raise CustomException(e, sys)

    def _preprocess_image(self, image_path: Path):

        try:
            logger.info(f"Preprocessing image: {image_path}")

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Image not found at {image_path}"
                )

            image = tf.keras.utils.load_img(
                image_path,
                target_size=self.config.image_size,
                color_mode="rgb"
            )

            image_array = tf.keras.utils.img_to_array(image)
            image_array = np.expand_dims(image_array, axis=0)

            logger.info(
                "Image loading and resizing completed. "
                "Model will perform preprocessing internally."
            )

            return image_array

        except Exception as e:
            logger.exception("Image preprocessing failed.")
            raise CustomException(e, sys)

    def predict(self, image_array):

        try:
            predictions = self.model.predict(
                image_array,
                verbose=0
            )

            predicted_class_id = int(
                np.argmax(predictions[0])
            )

            confidence = float(
                predictions[0][predicted_class_id]
            )

            predicted_class = self.config.class_names[
                predicted_class_id
            ]

            probabilities = {
                class_name: float(probability)
                for class_name, probability in zip(
                    self.config.class_names,
                    predictions[0]
                )
            }

            return {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "class_probabilities": probabilities
            }

        except Exception as e:
            logger.exception("Model prediction failed.")
            raise CustomException(e, sys)

    def initiate_model_inference(self, image_path: Path) -> dict:

        image_array = self._preprocess_image(image_path)

        return self.predict(image_array)