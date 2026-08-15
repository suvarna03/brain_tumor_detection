from brain_tumor_detection.entity.config_entity import ModelTrainingConfig


class ModelTrainer:

    def __init__(
        self,
        config: ModelTrainingConfig
    ):
        self.config = config

    def display_config(self):

        print("Model Training Configuration")
        print("-" * 40)

        print(
            f"Model Name      : {self.config.model_name}"
        )

        print(
            f"Image Size      : {self.config.image_size}"
        )

        print(
            f"Batch Size      : {self.config.batch_size}"
        )

        print(
            f"Epochs          : {self.config.epochs}"
        )

        print(
            f"Learning Rate   : {self.config.learning_rate}"
        )

        print(
            f"Include Top     : {self.config.include_top}"
        )

        print(
            f"Weights         : {self.config.weights}"
        )

        print(
            f"Classes         : {self.config.classes}"
        )

        print(
            f"Color Mode      : {self.config.color_mode}"
        )

        print(
            f"Normalize       : {self.config.normalize}"
        )