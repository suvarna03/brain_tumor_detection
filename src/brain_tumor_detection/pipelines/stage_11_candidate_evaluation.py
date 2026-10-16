
import sys

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

from tensorflow.keras.applications.resnet50 import preprocess_input

from brain_tumor_detection.config.configuration import ConfigurationManager
from brain_tumor_detection.exception import CustomException
from brain_tumor_detection.logger import logger


class CandidateEvaluationPipeline:

    def main(self):

        config = ConfigurationManager()

        inference_config = config.get_model_inference_config()
        validation_config = config.get_data_validation_config()

        test_dir = (
            validation_config.dataset_dir
            / "classification_task"
            / "test"
        )

        candidate_path = (
            inference_config.model_path.parent
            / "resnet50_candidate.keras"
        )

        class_names = inference_config.class_names

        if not candidate_path.is_file():
            raise FileNotFoundError(
                f"Candidate model not found: {candidate_path}"
            )

        if not inference_config.model_path.is_file():
            raise FileNotFoundError(
                f"Selected model not found: {inference_config.model_path}"
            )

        if not test_dir.is_dir():
            raise FileNotFoundError(
                f"Test directory not found: {test_dir}"
            )

        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            image_size=inference_config.image_size,
            batch_size=32,
            shuffle=False,
            color_mode="rgb"
        )

        if test_ds.class_names != class_names:
            raise ValueError(
                f"Expected classes {class_names}, "
                f"found {test_ds.class_names}"
            )

        y_true = np.concatenate(
            [labels.numpy() for _, labels in test_ds]
        )

        custom_objects = {
        "preprocess_input": preprocess_input
            }

        candidate_model = tf.keras.models.load_model(
            candidate_path,
            custom_objects=custom_objects,
            compile=False,
            safe_mode=True
        )

        selected_model = tf.keras.models.load_model(
            inference_config.model_path,
            custom_objects=custom_objects,
            compile=False,
            safe_mode=True
        )
        y_candidate = np.argmax(
            candidate_model.predict(test_ds, verbose=1),
            axis=1
        )

        y_selected = np.argmax(
            selected_model.predict(test_ds, verbose=1),
            axis=1
        )

        for model_name, predictions in [
            ("Candidate", y_candidate),
            ("Selected", y_selected)
        ]:

            print(f"\n{model_name} Model Results")
            print("=" * 40)

            print(
                "Accuracy:",
                round(accuracy_score(y_true, predictions), 4)
            )

            print(
                "Macro F1:",
                round(
                    f1_score(
                        y_true,
                        predictions,
                        average="macro"
                    ),
                    4
                )
            )

            print("\nClassification Report:")
            print(
                classification_report(
                    y_true,
                    predictions,
                    labels=list(range(len(class_names))),
                    target_names=class_names,
                    zero_division=0
                )
            )

            print("Confusion Matrix:")
            print(confusion_matrix(
                y_true,
                predictions,
                labels=list(range(len(class_names)))
            ))

            tumor_to_no_tumor = sum(
                (y_true != class_names.index("no_tumor"))
                & (predictions == class_names.index("no_tumor"))
            )

            print(
                "Tumor-to-no_tumor errors:",
                int(tumor_to_no_tumor)
            )


if __name__ == "__main__":

    try:
        logger.info(
            ">>>>>> Stage 11 Candidate Evaluation Started <<<<<<"
        )

        pipeline = CandidateEvaluationPipeline()
        pipeline.main()

        logger.info(
            ">>>>>> Stage 11 Candidate Evaluation Completed <<<<<<"
        )

    except Exception as e:
        logger.exception(e)
        raise CustomException(e, sys)
