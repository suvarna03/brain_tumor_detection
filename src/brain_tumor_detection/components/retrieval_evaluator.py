import sys
from pathlib import Path
from collections import Counter
import tensorflow as tf

import numpy as np
import pandas as pd

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
        random_seed=42):

        self.query_feature_extractor = (query_feature_extractor)

        self.query_pca = query_pca

        self.similarity_search = (similarity_search)

        self.model_config = model_config

        self.class_names = class_names

        self.test_dir = Path(test_dir)

        self.samples_per_class = (samples_per_class)

        self.top_k = top_k

        self.random_seed = random_seed

    def _get_test_images(self):

        try:

            logger.info(
                f"Scanning test directory: "
                f"{self.test_dir}")

            rng = np.random.default_rng(self.random_seed)

            selected_images = []

            for class_name in self.class_names:

                class_dir = (self.test_dir / class_name)

                if not class_dir.exists():

                    raise FileNotFoundError(
                        f"Test class directory not found: "
                        f"{class_dir}")

                image_paths = [path
                    for path in class_dir.iterdir()
                    if (path.is_file()
                        and path.suffix.lower()
                        in {".jpg", ".jpeg", ".png"})
                    ]

                if len(image_paths) < self.samples_per_class:

                    raise ValueError(
                        f"Class '{class_name}' has only "
                        f"{len(image_paths)} images. "
                        f"Required: "
                        f"{self.samples_per_class}")

                image_paths = sorted(image_paths)

                selected_indices = rng.choice(
                    len(image_paths),
                    size=self.samples_per_class,
                    replace=False)

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

            raise CustomException(e,sys)

    def _load_image(self,
        image_path):

        image = tf.keras.utils.load_img(
            image_path,
            target_size=(
                self.model_config.image_size
            ),
            color_mode="rgb"
        )

        image_array = (
            tf.keras.utils.img_to_array(
                image)
            )

        image_array = np.expand_dims(
            image_array,
            axis=0)

        return image_array

    def evaluate(self):

        try:

            selected_images = (
                self._get_test_images())

            results = []

            for query_class, image_path in selected_images:

                logger.info(
                    f"Evaluating query: "
                    f"{image_path}")

                image_array = (self._load_image(
                        image_path)
                    )

                query_features = (self.query_feature_extractor.extract(
                        image_array)
                    )
                
                reduced_query = (self.query_pca.transform(
                        query_features)
                    )

                retrieval_results = (self.similarity_search.search(
                        query_embedding=reduced_query,
                        top_k=self.top_k)
                    )

                retrieved_labels = [result["label"]
                    for result in retrieval_results
                    ]

                retrieved_class_names = [self.class_names[label]
                    for label in retrieved_labels
                    ]

                true_label = (self.class_names.index(
                        query_class)
                    )

                top1_correct = (
                    retrieved_labels[0] == true_label )

                label_counts = Counter(retrieved_labels)

                majority_label = (label_counts.most_common(1)[0][0])

                top5_majority_correct = (
                    majority_label == true_label )

                correct_retrievals = sum(
                    label == true_label
                    for label in retrieved_labels)

                top5_purity = (
                    correct_retrievals / self.top_k)


                results.append({

                    "query_class":
                        query_class,

                    "query_image":
                        str(image_path),

                    "top1_label":
                        retrieved_class_names[0],

                    "top1_similarity":
                        retrieval_results[0][
                            "similarity_score"
                        ],

                    "top1_correct":
                        top1_correct,

                    "top5_labels":
                        retrieved_class_names,

                    "top5_majority_label":
                        self.class_names[
                            majority_label
                        ],

                    "top5_majority_correct":
                        top5_majority_correct,

                    "top5_correct_count":
                        correct_retrievals,

                    "top5_purity":
                        top5_purity
                })

            return results

        except Exception as e:

            logger.exception(
                "Retrieval evaluation failed."
            )

            raise CustomException(e,sys)

    def summarize(self,
        results):

        dataframe = pd.DataFrame(results)

        top1_accuracy = (
            dataframe["top1_correct"].mean())

        top5_majority_accuracy = (
            dataframe["top5_majority_correct"].mean())

        mean_top5_purity = (
            dataframe["top5_purity"].mean())

        print("\nRetrieval Evaluation Results")

        print("============================")

        print(
            f"Total queries: "
            f"{len(dataframe)}")

        print(
            f"Top-1 retrieval accuracy: "
            f"{top1_accuracy:.4f}")

        print(
            f"Top-1 retrieval accuracy (%): "
            f"{top1_accuracy * 100:.2f}%")

        print(
            f"Top-5 majority accuracy: "
            f"{top5_majority_accuracy:.4f}")

        print(
            f"Top-5 majority accuracy (%): "
            f"{top5_majority_accuracy * 100:.2f}%")

        print(
            f"Mean Top-5 class purity: "
            f"{mean_top5_purity:.4f}")

        print(
            f"Mean Top-5 class purity (%): "
            f"{mean_top5_purity * 100:.2f}%")

        print("\nClass-wise Top-1 Accuracy" )

        print("-------------------------")

        class_summary = (
            dataframe.groupby("query_class")["top1_correct"].mean())

        for class_name in self.class_names:

            accuracy = (
                class_summary.get(class_name,0)
                )

            print(
                f"{class_name}: "
                f"{accuracy * 100:.2f}%"
            )

        return dataframe