from pathlib import Path

import numpy as np
from PIL import Image

from pathlib import Path

import numpy as np
from PIL import Image

from brain_tumor_detection.config.configuration import (
    ConfigurationManager
)

from brain_tumor_detection.components.query_feature_extractor import (
    QueryFeatureExtractor
)

from brain_tumor_detection.components.query_pca import (
    QueryPCA
)

from brain_tumor_detection.components.similarity_search import (
    SimilaritySearch
)


class QuerySimilarity:

    def main(self):

        print("Query Similarity Started")

        config_manager = ConfigurationManager()

        model_config = (
            config_manager.get_model_inference_config())

        faiss_config = (
            config_manager.get_faiss_config())

    
        query_image = Path(
            "artifacts/data_ingestion/"
            "brisc2025/brisc2025/"
            "classification_task/test/"
            "glioma/"
            "brisc2025_test_00001_gl_ax_t1.jpg")

        print("Query image:",
            query_image)

        image = Image.open(query_image).convert("RGB")

        image = image.resize(
            tuple(model_config.image_size))

        image_array = np.asarray(image,
            dtype=np.float32)

        image_array = np.expand_dims(image_array,
            axis=0)

        print(
            "Query image shape:",
            image_array.shape)

        feature_extractor = (QueryFeatureExtractor(
                config=model_config))

        query_features = (feature_extractor.extract(
                image_array)
        )

        print(
            "Query feature shape:",
            query_features.shape)


        pca = QueryPCA(pca_path=Path(
                "artifacts/pca/pca.pkl")
            )

        reduced_query = pca.transform(query_features)

        print(
            "Reduced query shape:",
            reduced_query.shape)

        similarity_search = (SimilaritySearch(
                config=faiss_config)
                )

        results = similarity_search.search(
            query_embedding=reduced_query,
            top_k=faiss_config.top_k)


        print("\nTop similar images:")

        for result in results:

            print(
                f"Rank: {result['rank']} | "
                f"Similarity: "
                f"{result['similarity_score']:.4f} | "
                f"Label: {result['label']} | "
                f"Path: {result['image_path']}"
            )

        print("\nQuery Similarity Completed")


if __name__ == "__main__":

    stage = QuerySimilarity()

    stage.main()