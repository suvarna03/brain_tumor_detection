from brain_tumor_detection.config.configuration import (
    ConfigurationManager
)

from brain_tumor_detection.components.model_inference import (
    ModelInference
)

from brain_tumor_detection.logger import logger
from brain_tumor_detection.exception import CustomException
import sys


class ModelInferencePipeline:

    def main(self):

        config = ConfigurationManager()

        model_inference_config = (
            config.get_model_inference_config()
        )

        model_inference = ModelInference(
            config=model_inference_config
        )

        image_path = input(
            "\nEnter the path of the MRI image: "
        ).strip()

        model_inference.initiate_model_inference(
            image_path=image_path
        )


if __name__ == "__main__":

    try:

        logger.info(
            ">>>>>> Stage 04 Model Inference Started <<<<<<"
        )

        obj = ModelInferencePipeline()

        obj.main()

        logger.info(
            ">>>>>> Stage 04 Model Inference Completed <<<<<<\n"
        )

    except Exception as e:
        logger.exception(e)
        raise CustomException(e, sys)