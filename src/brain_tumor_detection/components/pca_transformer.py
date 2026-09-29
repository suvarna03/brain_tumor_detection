import sys
import pickle
from pathlib import Path

import numpy as np

from sklearn.decomposition import PCA

from brain_tumor_detection.entity.config_entity import (
    PCAConfig
)

from brain_tumor_detection.utils.common import (
    create_directory
)

from brain_tumor_detection.logger import logger
from brain_tumor_detection.exception import CustomException


class PCATransformer:

    def __init__(self,config: PCAConfig):

        self.config = config

        logger.info(
            "Initializing PCATransformer.")

        create_directory(
            self.config.root_dir)

    def _load_features(self):

        try:

            logger.info(
                f"Loading feature data from "
                f"{self.config.input_feature_file}")

            if not self.config.input_feature_file.exists():

                raise FileNotFoundError(
                    f"Feature file not found at "
                    f"{self.config.input_feature_file}")

            with open(self.config.input_feature_file,"rb") as file:

                feature_data = pickle.load(file)

            features = feature_data["features"]
            labels = feature_data["labels"]
            image_paths = feature_data["image_paths"]

            logger.info(
                f"Loaded features shape: "
                f"{features.shape}")

            logger.info(
                f"Loaded labels shape: "
                f"{labels.shape}")

            logger.info(
                f"Loaded image paths count: "
                f"{len(image_paths)}")

            return (
                features,
                labels,
                image_paths)

        except Exception as e:

            logger.exception(
                "Failed to load feature data.")

            raise CustomException(e,sys)

    def _fit_pca(self,features):

        try:

            logger.info(
                "Starting PCA fitting.")

            logger.info(
                f"Input feature shape: "
                f"{features.shape}")

            logger.info(
                f"Number of PCA components: "
                f"{self.config.n_components}")

            pca = PCA(n_components=self.config.n_components,
                    random_state=42)

            reduced_features = pca.fit_transform(features)

            logger.info(
                "PCA fitting completed.")

            logger.info(
                f"Reduced feature shape: "
                f"{reduced_features.shape}")

            logger.info(
                f"Explained variance ratio sum: "
                f"{pca.explained_variance_ratio_.sum():.4f}")

            return (pca,
                reduced_features)

        except Exception as e:

            logger.exception(
                "PCA fitting failed.")

            raise CustomException(e,sys)

    def _save_pca_model(self,pca):

        try:

            logger.info(
                f"Saving PCA model to "
                f"{self.config.pca_model_file}")

            create_directory(
                self.config.root_dir)

            with open(self.config.pca_model_file,"wb") as file:

                pickle.dump(pca,file)

            logger.info(
                "PCA model saved successfully.")

        except Exception as e:

            logger.exception(
                "Failed to save PCA model.")

            raise CustomException(e,sys)

    def _save_reduced_features(self,reduced_features,labels,image_paths):

        try:

            logger.info(
                f"Saving reduced features to "
                f"{self.config.reduced_feature_file}")

            create_directory(self.config.root_dir)

            reduced_feature_data = {

                "features": reduced_features,

                "labels": labels,

                "image_paths": image_paths
            }

            with open(self.config.reduced_feature_file,"wb") as file:

                pickle.dump(reduced_feature_data,file)

            logger.info(
                "Reduced feature data saved successfully.")

        except Exception as e:

            logger.exception(
                "Failed to save reduced feature data.")

            raise CustomException(e,sys)

    def initiate_pca(self):

        try:

            (
                features,
                labels,
                image_paths
            ) = self._load_features()

            (
                pca,
                reduced_features
            ) = self._fit_pca(
                features
            )

            self._save_pca_model(
                pca
            )

            self._save_reduced_features(
                reduced_features,
                labels,
                image_paths
            )

            return {
                "pca": pca,

                "features": reduced_features,

                "labels": labels,

                "image_paths": image_paths
            }

        except Exception as e:

            logger.exception(
                "PCA transformation failed."
            )

            raise CustomException(e,sys)