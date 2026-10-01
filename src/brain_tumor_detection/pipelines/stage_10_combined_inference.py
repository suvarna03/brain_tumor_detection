from pathlib import Path

import numpy as np
from PIL import Image

from brain_tumor_detection.config.configuration import (
    ConfigurationManager
)

from brain_tumor_detection.components.model_inference import (
    ModelInference
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


class CombinedInferencePipeline:

    def main(self):

        print("Stage 10 Combined Inference Started")

        config_manager = (ConfigurationManager())

        model_config = (
            config_manager.get_model_inference_config()
            )

        faiss_config = (
            config_manager.get_faiss_config()
            )

        query_image = Path(
            input("\nEnter the path of the MRI image: ").strip())

        if not query_image.exists():

            raise FileNotFoundError(
                f"Query image not found: "
                f"{query_image}"
            )

        print(
            "\nQuery image:",
            query_image)

        
        ### PART 1 — CLASSIFICATION
       

        print("\nRunning model inference...")

        model_inference = (ModelInference(config=model_config))

        classification_result = (
            model_inference.initiate_model_inference(
                image_path=query_image)
            )
        
        ### PART 2 — LOAD QUERY IMAGE
        
        image = (
            Image.open(query_image).convert("RGB")
            )

        image = image.resize(tuple(model_config.image_size))

        image_array = np.asarray(
            image,
            dtype=np.float32)

        image_array = np.expand_dims(
            image_array,
            axis=0)

        print(
            "Query image shape:",
            image_array.shape
        )

      
    ### PART 3 — FEATURE EXTRACTION
       

        print("\nExtracting query features...")

        query_feature_extractor = (
            QueryFeatureExtractor(config=model_config)
        )

        query_features = (query_feature_extractor.extract(image_array)
        )

        print(
            "Query feature shape:",
            query_features.shape
        )

       
        ### PART 4 — PCA
        

        print(
            "\nApplying saved PCA..."
        )

        query_pca = QueryPCA(
            pca_path=Path(
                "artifacts/pca/pca.pkl"
            )
        )

        reduced_query = (
            query_pca.transform(
                query_features
            )
        )

        print(
            "Reduced query shape:",
            reduced_query.shape
        )

        ### PART 5 — FAISS SEARCH
        

        print("\nSearching similar images...")

        similarity_search = (SimilaritySearch(
                config=faiss_config)
        )

        similarity_results = (
            similarity_search.search(
                query_embedding=reduced_query,
                top_k=faiss_config.top_k
            )
        )

        
        ### FINAL RESULT
      

        final_result = {

            "classification":
                classification_result,

            "similar_images":
                similarity_results
        }

        ### DISPLAY CLASSIFICATION
        

        print("\n================================")

        print("CLASSIFICATION RESULT")

        print("================================")

        print(
            "Predicted class:",
            classification_result["predicted_class"]
        )

        print(
            "Confidence:",
            f"{classification_result['confidence'] * 100:.2f}%"
        )

        print(
            "\nClass probabilities:"
        )

        for (
            class_name,
            probability
        ) in classification_result[
            "probabilities"
        ].items():

            print(
                f"{class_name}: "
                f"{probability * 100:.2f}%"
            )

        
        #### DISPLAY SIMILAR IMAGES
      
        print("\n================================")

        print("SIMILAR REFERENCE IMAGES")

        print("================================")

        for result in similarity_results:

            print(
                f"Rank: "
                f"{result['rank']} | "
                f"Similarity: "
                f"{result['similarity_score']:.4f} | "
                f"Label: "
                f"{result['label']} | "
                f"Path: "
                f"{result['image_path']}"
            )

        print("\nStage 10 Combined Inference Completed")

        return final_result


if __name__ == "__main__":

    stage = (
        CombinedInferencePipeline()
    )

    stage.main()