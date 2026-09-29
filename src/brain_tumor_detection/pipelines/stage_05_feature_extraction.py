import sys

from brain_tumor_detection.config.configuration import (
    ConfigurationManager
)

from brain_tumor_detection.components.feature_extractor import (
    FeatureExtractor
)

from brain_tumor_detection.logger import logger
from brain_tumor_detection.exception import CustomException


class FeatureExtractionTrainingPipeline:

    def __init__(self):
        pass

    def main(self):

        config = ConfigurationManager()

        feature_extraction_config = (config.get_feature_extraction_config())

        feature_extractor = FeatureExtractor(config=feature_extraction_config)

        feature_data = (feature_extractor.initiate_feature_extraction())

        print("\nFeature Extraction Completed")

        print("-----------------------------")

        print(
            "Features shape:",
            feature_data["features"].shape)

        print(
            "Labels shape:",
            feature_data["labels"].shape)

        print(
            "Image paths:",
            len(feature_data["image_paths"]))

        print(
            "Feature file:",
            feature_extraction_config.feature_file)


if __name__ == "__main__":

    try:

        logger.info(
            ">>>>>> Stage 05 Feature Extraction "
            "Started <<<<<<")

        obj = FeatureExtractionTrainingPipeline()

        obj.main()

        logger.info(
            ">>>>>> Stage 05 Feature Extraction "
            "Completed <<<<<<\n")

    except Exception as e:

        logger.exception(e)

        raise CustomException(e,sys)