import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import pickle

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
        logger.info("Initializing FeatureExtractor.")

        create_directory(
            self.config.root_dir)

        self.model_path = Path(
            "artifacts/model_trainer/resnet50_finetuned.keras")

        self.model = self._load_model()

        self.feature_model = (
            self._build_feature_model())

    def _load_model(self):

        try:

            logger.info(
                f"Loading trained model from "
                f"{self.model_path}")

            if not self.model_path.exists():

                raise FileNotFoundError(
                    f"Model not found at "
                    f"{self.model_path}")

            model = load_model(
                self.model_path,
                custom_objects={
                    "preprocess_input":
                    preprocess_input},
                compile=False)

            logger.info(
                "Trained model loaded successfully.")

            return model

        except Exception as e:

            logger.exception(
                "Failed to load trained model.")

            raise CustomException(e,sys)

    def _build_feature_model(self):

        try:

            logger.info(
                "Building feature extraction model.")

            dummy_input = tf.zeros(
                shape=(1,
                    self.config.image_size[0],
                    self.config.image_size[1],
                    3),
                dtype=tf.float32)

            self.model(
                dummy_input,
                training=False)

            logger.info(
                "Trained model initialized successfully.")

            feature_layer = (
                self.model.get_layer(
                    "global_average_pooling2d")
                )

            feature_model = tf.keras.Model(
                inputs=self.model.inputs,
                outputs=feature_layer.output)

            logger.info(
                "Feature extraction model "
                "created successfully.")

            logger.info(
                f"Feature model output shape: "
                f"{feature_model.output_shape}")

            return feature_model

        except Exception as e:

            logger.exception(
                "Failed to build feature extraction model.")

            raise CustomException(e,sys)

    def _preprocess_image(self,image_path: Path):

        try:

            image = tf.keras.utils.load_img(
                image_path,
                target_size=self.config.image_size,
                color_mode="rgb")

            image_array = (
                tf.keras.utils.img_to_array(
                    image)
                    )

            return image_array

        except Exception as e:

            logger.exception(
                f"Failed to preprocess image: "
                f"{image_path}")

            raise CustomException(e,sys)

    def _get_training_images(self):

        try:

            logger.info(
                f"Scanning training directory: "
                f"{self.config.train_dir}")

            if not self.config.train_dir.exists():

                raise FileNotFoundError(
                    f"Training directory not found at "
                    f"{self.config.train_dir}")

            image_paths = []

            allowed_extensions = {
                ".jpg",
                ".jpeg",
                ".png"}

            for class_name in self.config.class_names:

                class_dir = (
                    self.config.train_dir
                    / class_name)

                if not class_dir.exists():

                    raise FileNotFoundError(
                        f"Class directory not found: "
                        f"{class_dir}")

                for image_path in sorted(class_dir.iterdir()):

                    if (image_path.is_file()
                        and image_path.suffix.lower()
                        in allowed_extensions):

                        image_paths.append(image_path)

            logger.info(
                f"Total training images found: "
                f"{len(image_paths)}")

            return image_paths

        except Exception as e:

            logger.exception(
                "Failed to collect training images.")

            raise CustomException(e,sys)


    def extract_dataset_features(self):

        try:

            logger.info(
                "Starting full dataset feature extraction.")

            image_paths = (
                self._get_training_images())

            if len(image_paths) == 0:

                raise ValueError(
                    "No training images found.")

            all_embeddings = []
            all_labels = []
            all_paths = []

            batch_size = (
                self.config.batch_size
            )

            total_images = len(image_paths)

            for start_index in range(
                0,
                total_images,
                batch_size):

                end_index = min(
                    start_index + batch_size,
                    total_images)

                batch_paths = image_paths[start_index:end_index ]

                logger.info(
                    f"Processing images "
                    f"{start_index + 1} "
                    f"to {end_index} "
                    f"of {total_images}")

                batch_images = []

                batch_labels = []

                for image_path in batch_paths:

                    image_array = (
                        self._preprocess_image(
                            image_path)
                        )

                    batch_images.append(
                        image_array)

                    class_name = (
                        image_path.parent.name)

                    label = (
                        self.config.class_names.index(class_name))

                    batch_labels.append(label)

                batch_images = np.asarray(batch_images,
                    dtype=np.float32)

                batch_labels = np.asarray(batch_labels,
                    dtype=np.int32)

                batch_embeddings = (
                    self.feature_model.predict(
                        batch_images,
                        verbose=0))

                batch_embeddings = np.asarray(
                    batch_embeddings,
                    dtype=np.float32
                )

                all_embeddings.append(batch_embeddings)

                all_labels.append(batch_labels)

                all_paths.extend(
                    [str(path) for path in batch_paths]
                    )

            features = np.vstack(all_embeddings)

            labels = np.concatenate(all_labels)

            image_paths = np.asarray(all_paths)

            logger.info(f"Final feature matrix shape: "
                f"{features.shape}")

            logger.info(f"Final labels shape: "
                f"{labels.shape}")

            logger.info(f"Final image paths count: "
                f"{len(image_paths)}")

            return {
                "features": features,
                "labels": labels,
                "image_paths": image_paths
            }

        except Exception as e:

            logger.exception(
                "Dataset feature extraction failed."
            )

            raise CustomException(e,sys)

    def save_features(self,feature_data: dict):

        try:

            logger.info(
                f"Saving extracted features to "
                f"{self.config.feature_file}")

            create_directory(self.config.root_dir)

            with open(self.config.feature_file,
                "wb") as file:

                pickle.dump(
                    feature_data,
                    file)

            logger.info(
                "Feature data saved successfully.")

        except Exception as e:

            logger.exception(
                "Failed to save feature data.")

            raise CustomException(e,sys)

    def initiate_feature_extraction(self):

        feature_data = (
            self.extract_dataset_features())

        self.save_features(feature_data)

        return feature_data