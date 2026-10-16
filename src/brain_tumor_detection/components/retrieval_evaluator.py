
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

from brain_tumor_detection.exception import CustomException
from brain_tumor_detection.logger import logger


class RetrievalEvaluator:

    def __init__(
        self,
        query_feature_extractor,
        query_pca,
        similarity_search,
        model_config,
        class_names,
        test_dir,
        samples_per_class=20,
        top_k=5,
        random_seed=42
    ):

        self.query_feature_extractor = query_feature_extractor
        self.query_pca = query_pca
        self.similarity_search = similarity_search
        self.model_config = model_config
        self.class_names = class_names
        self.test_dir = Path(test_dir)
        self.samples_per_class = samples_per_class
        self.top_k = top_k
        self.random_seed = random_seed

    def _get_test_images(self):

        try:
            logger.info(
                f"Scanning test directory: {self.test_dir}"
            )

            if not self.test_dir.is_dir():
                raise FileNotFoundError(
                    f"Test directory not found: {self.test_dir}"
                )

            if self.samples_per_class < 1:
                raise ValueError(
                    "samples_per_class must be at least 1."
                )

            rng = np.random.default_rng(
                self.random_seed
            )

            selected_images = []

            for class_name in self.class_names:

                class_dir = self.test_dir / class_name

                if not class_dir.is_dir():
                    raise FileNotFoundError(
                        f"Test class directory not found: "
                        f"{class_dir}"
                    )

                image_paths = sorted([
                    path
                    for path in class_dir.iterdir()
                    if (
                        path.is_file()
                        and path.suffix.lower()
                        in {".jpg", ".jpeg", ".png"}
                    )
                ])

                if len(image_paths) < self.samples_per_class:
                    raise ValueError(
                        f"Class '{class_name}' has only "
                        f"{len(image_paths)} images. "
                        f"Required: {self.samples_per_class}"
                    )

                selected_indices = rng.choice(
                    len(image_paths),
                    size=self.samples_per_class,
                    replace=False
                )

                for index in selected_indices:
                    selected_images.append(
                        (
                            class_name,
                            image_paths[index]
                        )
                    )

            logger.info(
                f"Selected {len(selected_images)} "
                f"test images for retrieval evaluation."
            )

            return selected_images

        except Exception as e:
            logger.exception(
                "Failed to collect test images."
            )
            raise CustomException(e, sys)

    def _load_image(self, image_path):

        image = tf.keras.utils.load_img(
            image_path,
            target_size=self.model_config.image_size,
            color_mode="rgb"
        )

        image_array = tf.keras.utils.img_to_array(image)

        return np.expand_dims(
            image_array,
            axis=0
        )

    def evaluate(self):

        try:
            if self.top_k < 1:
                raise ValueError(
                    "top_k must be at least 1."
                )

            selected_images = self._get_test_images()

            results = []

            for query_class, image_path in selected_images:

                logger.info(
                    f"Evaluating query: {image_path}"
                )

                image_array = self._load_image(
                    image_path
                )

                query_features = (
                    self.query_feature_extractor.extract(
                        image_array
                    )
                )

                reduced_query = self.query_pca.transform(
                    query_features
                )

                retrieval_results = (
                    self.similarity_search.search(
                        query_embedding=reduced_query,
                        top_k=self.top_k
                    )
                )

                if not retrieval_results:
                    raise ValueError(
                        f"No images retrieved for query: "
                        f"{image_path}"
                    )

                retrieved_labels = [
                    result["label"]
                    for result in retrieval_results
                ]

                if any(
                    not isinstance(label, (int, np.integer))
                    or label < 0
                    or label >= len(self.class_names)
                    for label in retrieved_labels
                ):
                    raise ValueError(
                        "Retrieved labels do not match "
                        "the configured classes."
                    )

                retrieved_class_names = [
                    self.class_names[label]
                    for label in retrieved_labels
                ]

                true_label = self.class_names.index(
                    query_class
                )

                top1_correct = (
                    retrieved_labels[0] == true_label
                )

                label_counts = Counter(
                    retrieved_labels
                )

                majority_label = (
                    label_counts.most_common(1)[0][0]
                )

                top_k_majority_correct = (
                    majority_label == true_label
                )

                retrieved_count = len(retrieval_results)

                correct_retrievals = sum(
                    label == true_label
                    for label in retrieved_labels
                )

                top_k_purity = (
                    correct_retrievals / retrieved_count
                )

                results.append({
                    "query_class": query_class,
                    "query_image": str(image_path),
                    "top1_label": retrieved_class_names[0],
                    "top1_similarity": retrieval_results[0][
                        "similarity_score"
                    ],
                    "top1_correct": top1_correct,
                    "top_k_labels": retrieved_class_names,
                    "top_k_majority_label": self.class_names[
                        majority_label
                    ],
                    "top_k_majority_correct": top_k_majority_correct,
                    "top_k_correct_count": correct_retrievals,
                    "top_k_purity": top_k_purity,
                    "retrieved_count": retrieved_count
                })

            return results

        except Exception as e:
            logger.exception(
                "Retrieval evaluation failed."
            )
            raise CustomException(e, sys)

    def summarize(self, results):

        if not results:
            raise ValueError(
                "Cannot summarize retrieval evaluation: "
                "no results found."
            )

        dataframe = pd.DataFrame(results)

        top1_accuracy = dataframe[
            "top1_correct"
        ].mean()

        top_k_majority_accuracy = dataframe[
            "top_k_majority_correct"
        ].mean()

        mean_top_k_purity = dataframe[
            "top_k_purity"
        ].mean()

        print("\nRetrieval Evaluation Results")
        print("============================")

        print(
            f"Total queries: {len(dataframe)}"
        )

        print(
            f"Top-1 retrieval accuracy: "
            f"{top1_accuracy:.4f}"
        )

        print(
            f"Top-1 retrieval accuracy (%): "
            f"{top1_accuracy * 100:.2f}%"
        )

        print(
            f"Top-{self.top_k} majority accuracy: "
            f"{top_k_majority_accuracy:.4f}"
        )

        print(
            f"Top-{self.top_k} majority accuracy (%): "
            f"{top_k_majority_accuracy * 100:.2f}%"
        )

        print(
            f"Mean Top-{self.top_k} class purity: "
            f"{mean_top_k_purity:.4f}"
        )

        print(
            f"Mean Top-{self.top_k} class purity (%): "
            f"{mean_top_k_purity * 100:.2f}%"
        )

        print("\nClass-wise Top-1 Accuracy")
        print("-------------------------")

        class_summary = dataframe.groupby(
            "query_class"
        )["top1_correct"].mean()

        for class_name in self.class_names:

            accuracy = class_summary.get(
                class_name,
                0
            )

            print(
                f"{class_name}: "
                f"{accuracy * 100:.2f}%"
            )

        return dataframe
