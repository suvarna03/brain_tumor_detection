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


class Stage08QuerySimilarity:

    def main(self):

        print("Stage 08 Query Similarity Search Started")

        config_manager = ConfigurationManager()

        model_config = config_manager.get_model_inference_config()
        faiss_config = config_manager.get_faiss_config()

        pca_path = Path("artifacts/pca/pca.pkl")

        query_image = Path(
            "artifacts/data_ingestion/"
            "brisc2025/brisc2025/"
            "classification_task/test/glioma/"
            "brisc2025_test_00001_gl_ax_t1.jpg"
        )

        query_feature_extractor = QueryFeatureExtractor(
            model_config
        )

        query_pca = QueryPCA(pca_path)

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
            reduced_features,
            top_k=faiss_config.top_k
        )

        print("\nQuery image:")
        print(query_image)

        print("\nSimilar images:")

        for result in results:
            print(
                f"Rank: {result['rank']} | "
                f"Similarity: {result['similarity_score']:.4f} | "
                f"Label ID: {result['label']} | "
                f"Image: {result['image_path']}"
            )

        print("\nStage 08 Query Similarity Search Completed")


if __name__ == "__main__":
    stage = Stage08QuerySimilarity()
    stage.main()