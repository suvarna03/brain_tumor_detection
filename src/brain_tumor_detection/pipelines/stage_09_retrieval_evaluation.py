from pathlib import Path

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

from brain_tumor_detection.components.retrieval_evaluator import (
    RetrievalEvaluator
)


class RetrievalEvaluationPipeline:
    """
    Retrieval evaluation on 80 randomly sampled unseen test images (20 per class), using fixed random seed 42.
    """

    def main(self):

        print(
            "Stage 09 Retrieval Evaluation Started"
        )
        config_manager = (
            ConfigurationManager()
        )

        model_config = (
            config_manager
            .get_model_inference_config()
        )

        faiss_config = (
            config_manager
            .get_faiss_config()
        )

        class_names = [
            "glioma",
            "meningioma",
            "no_tumor",
            "pituitary"
        ]

        test_dir = Path(
            "artifacts/data_ingestion/"
            "brisc2025/brisc2025/"
            "classification_task/test"
        )

        query_feature_extractor = (
            QueryFeatureExtractor(
                config=model_config
            )
        )

        query_pca = QueryPCA(
            pca_path=Path(
                "artifacts/pca/pca.pkl"
            )
        )

        similarity_search = (
            SimilaritySearch(
                config=faiss_config
            )
        )


        evaluator = RetrievalEvaluator(

            query_feature_extractor=(
                query_feature_extractor
            ),

            query_pca=query_pca,

            similarity_search=(
                similarity_search
            ),

            model_config=model_config,

            class_names=class_names,

            test_dir=test_dir,

            samples_per_class=20,

            top_k=5,

            random_seed=42
        )


        results = (
            evaluator.evaluate()
        )

        dataframe = (
            evaluator.summarize(
                results
            )
        )

        print("\nFirst 10 evaluation results:")

        print(
            dataframe[
                [
                    "query_class",
                    "top1_label",
                    "top1_similarity",
                    "top1_correct",
                    "top5_majority_label",
                    "top5_majority_correct",
                    "top5_purity"
                ]].head(10).to_string(index=False)
        )

        print("\nStage 09 Retrieval Evaluation Completed")


if __name__ == "__main__":

    stage = (
        RetrievalEvaluationPipeline()
    )

    stage.main()