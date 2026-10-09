import shutil
import sys
from pathlib import Path

import kagglehub

from brain_tumor_detection.entity.config_entity import DataIngestionConfig
from brain_tumor_detection.exception import CustomException
from brain_tumor_detection.logger import logger
from brain_tumor_detection.utils.common import create_directory


class DataIngestion:

    def __init__(self, config: DataIngestionConfig):
        self.config = config

        create_directory(self.config.root_dir)

    def download_dataset(self) -> Path:
        try:
            logger.info("Entered download_dataset()")

            if self.config.dataset_dir.exists():
                logger.info(
                    f"Dataset already exists at "
                    f"{self.config.dataset_dir}. Skipping download."
                )
            else:
                downloaded_path = Path(
                    kagglehub.dataset_download(
                        self.config.dataset_name
                    )
                )

                shutil.copytree(
                    downloaded_path,
                    self.config.dataset_dir
                )

                logger.info(
                    f"Dataset successfully downloaded and stored at "
                    f"{self.config.dataset_dir}"
                )

            return self.config.dataset_dir

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self) -> Path:
        return self.download_dataset()