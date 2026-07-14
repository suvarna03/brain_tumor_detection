import os
import sys
from brain_tumor_detection.entity.config_entity import DataValidationConfig
from brain_tumor_detection.utils.common import create_directory
from brain_tumor_detection.exception import CustomException
from brain_tumor_detection.logger import logger


class DataValidation:

    def __init__(
        self,
        config: DataValidationConfig):
        self.config = config

        create_directory(
            self.config.root_dir
            )

    def _validate_dataset_exists(self) -> tuple:

        if self.config.dataset_dir.exists():
            return True, ""

        return (
            False,
            f"Dataset directory does not exist: {self.config.dataset_dir}"
            )

    def _validate_folder_structure(self) -> tuple:

        classification_dir = (
            self.config.dataset_dir / "classification_task"
            )

        train_dir = classification_dir / "train"
        test_dir = classification_dir / "test"

        if not classification_dir.exists():
            return (
                False,
                "classification_task directory not found"
            )

        if not train_dir.exists():
            return (
                False,
                "train directory not found"
            )

        if not test_dir.exists():
            return (
                False,
                "test directory not found"
            )

        return True, ""

    def _validate_class_names(self) -> tuple:

        classification_dir = (
            self.config.dataset_dir / "classification_task")

        train_dir = classification_dir / "train"
        test_dir = classification_dir / "test"

        for cls in self.config.expected_classes:

            train_class_dir = train_dir / cls
            test_class_dir = test_dir / cls

            if not train_class_dir.exists():
                return (
                    False,
                    f"Missing class folder '{cls}' in train"
                )

            if not test_class_dir.exists():
                return (
                    False,
                    f"Missing class folder '{cls}' in test"
                )

        return True, ""

    def _validate_empty_images(self) -> tuple:

        classification_dir = (
            self.config.dataset_dir / "classification_task")

        train_dir = classification_dir / "train"
        test_dir = classification_dir / "test"

        empty_files = []

        for split_dir in [train_dir, test_dir]:

            for cls in self.config.expected_classes:

                class_dir = split_dir / cls

                for image_file in class_dir.iterdir():

                    if image_file.is_file():

                        if os.path.getsize(image_file) == 0:

                            empty_files.append(
                                str(image_file)
                            )

        if empty_files:

            return (
                False,
                "Empty images found:\n"
                + "\n".join(empty_files)
            )

        return True, ""

    def _write_validation_status(
        self,
        status: bool,
        message: str) -> None:

        with open(
            self.config.status_file,
            "w") as file:

            file.write(
                f"STATUS: {status}\n\n"
            )

            file.write(
                f"MESSAGE:\n{message}"
            )

    def initiate_data_validation(self) -> bool:

        try:

            checks = [
                self._validate_dataset_exists,
                self._validate_folder_structure,
                self._validate_class_names,
                self._validate_empty_images
            ]

            for check in checks:

                status, message = check()

                if not status:

                    self._write_validation_status(
                        False,
                        message
                    )

                    logger.error(message)

                    return False

            self._write_validation_status(
                True,
                "Validation completed successfully.")

            logger.info(
                "Data validation completed successfully.")

            return True

        except Exception as e:
            raise CustomException(e, sys)