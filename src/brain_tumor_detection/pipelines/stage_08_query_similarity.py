
import sys
from pathlib import Path

import tensorflow as tf

from brain_tumor_detection.components.query_feature_extractor import (
    QueryFeatureExtractor
)
from brain_tumor_detection.components.query_pca import QueryPCA
from brain_tumor_detection.components.similarity_search import (
    SimilaritySearch
)
from brain_tumor_detection.config.configuration import ConfigurationManager
from brain_tumor_detection.exception import CustomException
from brain_tumor_detection.logger import logger


class Stage08QuerySimilarity:

    def main(self):

        config_manager = ConfigurationManager()

        model_config = config_manager.get_model_inference_config()
        faiss_config = config_manager.get_faiss_config()
        pca_config = config_manager.get_pca_config()

        query_image = Path(
            input("Enter MRI image path: ").strip()
        )

        if not query_image.is_file():
            raise FileNotFoundError(
                f"Query image not found: {query_image}"
            )

        query_feature_extractor = QueryFeatureExtractor(
            model_config
        )

        query_pca = QueryPCA(
            pca_config.pca_model_file
        )

        similarity_search = SimilaritySearch(
            faiss_config
        )

        image = tf.keras.utils.load_img(
            query_image,
            target_size=model_config.image_size,
            color_mode="rgb"
        )

        image_array = tf.keras.utils.img_to_array(image)
        image_array = image_array[None, ...]

        query_features = query_feature_extractor.extract(
            image_array
        )

        reduced_features = query_pca.transform(
            query_features
        )

        results = similarity_search.search(
            query_embedding=reduced_features,
            top_k=faiss_config.top_k
        )

        if not results:
            raise ValueError(
                "No similar images were returned."
            )

        print("\nQuery image:")
        print(query_image)

        print("\nSimilar images:")

        for result in results:
            label_id = result["label"]

            if label_id < 0 or label_id >= len(model_config.class_names):
                raise ValueError(
                    f"Invalid class label returned: {label_id}"
                )

            class_name = model_config.class_names[label_id]

            print(
                f"Rank: {result['rank']} | "
                f"Similarity: {result['similarity_score']:.4f} | "
                f"Class: {class_name} | "
                f"Image: {result['image_path']}"
            )


if __name__ == "__main__":

    try:
        logger.info(
            ">>>>>> Stage 08 Query Similarity Search Started <<<<<<"
        )

        stage = Stage08QuerySimilarity()
        stage.main()

        logger.info(
            ">>>>>> Stage 08 Query Similarity Search Completed <<<<<<"
        )

    except Exception as e:
        logger.exception(e)
        raise CustomException(e, sys)
