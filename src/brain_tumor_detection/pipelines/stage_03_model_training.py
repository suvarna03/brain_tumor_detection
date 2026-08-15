from brain_tumor_detection.config.configuration import (
    ConfigurationManager
)

from brain_tumor_detection.components.model_trainer import (
    ModelTrainer
)


class ModelTrainingPipeline:

    def main(self):

        config = ConfigurationManager()

        model_training_config = (
            config.get_model_training_config()
        )

        model_trainer = ModelTrainer(
            config=model_training_config
        )

        model_trainer.display_config()


if __name__ == "__main__":

    obj = ModelTrainingPipeline()

    obj.main()