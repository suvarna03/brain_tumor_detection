from pathlib import Path

import tensorflow as tf

from brain_tumor_detection.components.model_inference import (
    ModelInference
)
from brain_tumor_detection.components.query_feature_extractor import (
    QueryFeatureExtractor
)
from brain_tumor_detection.components.query_pca import QueryPCA
from brain_tumor_detection.components.similarity_search import (
    SimilaritySearch
)
from brain_tumor_detection.config.configuration import ConfigurationManager


class Stage10CombinedInference:

    def main(self):

        print("Stage 10 Combined Inference Started")

        config_manager = ConfigurationManager()

        model_config = config_manager.get_model_inference_config()
        faiss_config = config_manager.get_faiss_config()

        pca_path = Path("artifacts/pca/pca.pkl")

        query_image = Path(
            input("Enter MRI image path: ").strip()
        )

        if not query_image.exists():
            print(f"Image not found: {query_image}")
            return

        model_inference = ModelInference(
            model_config
        )

        query_feature_extractor = QueryFeatureExtractor(
            model_config
        )

        query_pca = QueryPCA(pca_path)

        similarity_search = SimilaritySearch(
            faiss_config
        )

        classification_result = (
            model_inference.initiate_model_inference(
                query_image
            )
        )

        print("\nClassification Result")
        print("---------------------")

        print(
            f"Predicted class: "
            f"{classification_result['predicted_class']}"
        )

        print(
            f"Confidence: "
            f"{classification_result['confidence']:.4f}"
        )

        print("\nClass probabilities:")

        for class_name, probability in (
            classification_result["class_probabilities"].items()
        ):
            print(
                f"{class_name}: {probability:.4f}"
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

        similarity_results = similarity_search.search(
            query_embedding=reduced_features,
            top_k=faiss_config.top_k
        )

        print("\nSimilar Images")
        print("--------------")

        for result in similarity_results:

            label = model_config.class_names[
                result["label"]
            ]

            print(
                f"Rank: {result['rank']} | "
                f"Label: {label} | "
                f"Similarity: "
                f"{result['similarity_score']:.4f} | "
                f"Image: {result['image_path']}"
            )

        print("\nStage 10 Combined Inference Completed")


if __name__ == "__main__":
    stage = Stage10CombinedInference()
    stage.main()