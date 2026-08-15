from brain_tumor_detection.config.configuration import ConfigurationManager
from brain_tumor_detection.components.data_ingestion import DataIngestion
from brain_tumor_detection.logger import logger
from brain_tumor_detection.exception import CustomException
import sys


class DataIngestionTrainingPipeline:

    def __init__(self):
        pass

    def main(self):

        config = ConfigurationManager()

        data_ingestion_config = (
            config.get_data_ingestion_config()
        )

        data_ingestion = DataIngestion(
            config=data_ingestion_config
        )

        data_ingestion.initiate_data_ingestion()


if __name__ == "__main__":

    try:

        logger.info(
            ">>>>>> Stage 01 Data Ingestion Started <<<<<<"
        )

        obj = DataIngestionTrainingPipeline()

        obj.main()

        logger.info(
            ">>>>>> Stage 01 Data Ingestion Completed <<<<<<\n"
        )

    except Exception as e:
        logger.exception(e)
        raise CustomException(e, sys)