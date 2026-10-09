import sys

from brain_tumor_detection.components.data_validation import DataValidation
from brain_tumor_detection.config.configuration import ConfigurationManager
from brain_tumor_detection.exception import CustomException
from brain_tumor_detection.logger import logger


class DataValidationPipeline:

    def main(self):

        config = ConfigurationManager()

        data_validation_config = (
            config.get_data_validation_config()
        )

        data_validation = DataValidation(
            config=data_validation_config
        )

        status = data_validation.initiate_data_validation()

        if not status:
            raise RuntimeError(
                "Data validation failed. Check artifacts/data_validation/status.txt"
            )


if __name__ == "__main__":

    try:
        logger.info(
            ">>>>>> Stage 02 Data Validation Started <<<<<<"
        )

        obj = DataValidationPipeline()
        obj.main()

        logger.info(
            ">>>>>> Stage 02 Data Validation Completed <<<<<<\n"
        )

    except Exception as e:
        logger.exception(e)
        raise CustomException(e, sys)