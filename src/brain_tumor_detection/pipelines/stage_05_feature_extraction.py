from brain_tumor_detection.config.configuration import (
    ConfigurationManager
)

from brain_tumor_detection.components.feature_extractor import (
    FeatureExtractor
)

from brain_tumor_detection.logger import logger
from brain_tumor_detection.exception import CustomException

import sys


class FeatureExtractionTrainingPipeline:

    def __init__(self):
        pass

    def main(self):

        config = ConfigurationManager()

        feature_extraction_config = (
            config.get_feature_extraction_config()
        )

        feature_extractor = FeatureExtractor(
            config=feature_extraction_config
        )

        image_path = input(
            "\nEnter the path of the MRI image: "
        ).strip()

        embedding = (
            feature_extractor.initiate_feature_extraction(
                image_path=image_path
            )
        )

        print(
            "\nFeature Extraction Result"
        )

        print(
            "-------------------------"
        )

        print(
            "Embedding shape:",
            embedding.shape
        )

        print(
            "First 10 embedding values:",
            embedding[:10]
        )


if __name__ == "__main__":

    try:

        logger.info(
            ">>>>>> Stage 05 Feature Extraction Started <<<<<<"
        )

        obj = FeatureExtractionTrainingPipeline()

        obj.main()

        logger.info(
            ">>>>>> Stage 05 Feature Extraction Completed <<<<<<\n"
        )

    except Exception as e:

        logger.exception(e)

        raise CustomException(e,sys)