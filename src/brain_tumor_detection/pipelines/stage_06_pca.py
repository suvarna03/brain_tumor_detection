import sys

from brain_tumor_detection.config.configuration import (
    ConfigurationManager
)

from brain_tumor_detection.components.pca_transformer import (
    PCATransformer
)

from brain_tumor_detection.logger import logger
from brain_tumor_detection.exception import CustomException


class PCATrainingPipeline:

    def __init__(self):
        pass

    def main(self):

        config = ConfigurationManager()

        pca_config = (config.get_pca_config())

        pca_transformer = PCATransformer(config=pca_config)

        pca_data = (pca_transformer.initiate_pca())

        print("\nPCA Transformation Completed")

        print("-----------------------------")

        print("Original feature shape:",pca_data["pca"].n_features_in_)

        print("Reduced feature shape:",pca_data["features"].shape)

        print("Labels shape:",pca_data["labels"].shape)

        print("Image paths:",len(pca_data["image_paths"]))

        print("PCA components:",pca_config.n_components)

        print("Explained variance:",
            pca_data["pca"].explained_variance_ratio_.sum())

        print("PCA model:",pca_config.pca_model_file)

        print("Reduced feature file:",pca_config.reduced_feature_file)


if __name__ == "__main__":

    try:

        logger.info(
            ">>>>>> Stage 06 PCA Started <<<<<<")

        obj = PCATrainingPipeline()

        obj.main()

        logger.info(
            ">>>>>> Stage 06 PCA Completed <<<<<<\n")

    except Exception as e:

        logger.exception(e)

        raise CustomException(e,sys)