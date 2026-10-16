
import sys

from brain_tumor_detection.components.query_feature_extractor import (
    QueryFeatureExtractor
)
from brain_tumor_detection.components.query_pca import QueryPCA
from brain_tumor_detection.components.retrieval_evaluator import (
    RetrievalEvaluator
)
from brain_tumor_detection.components.similarity_search import (
    SimilaritySearch
)
from brain_tumor_detection.config.configuration import ConfigurationManager
from brain_tumor_detection.exception import CustomException
from brain_tumor_detection.logger import logger


class Stage09RetrievalEvaluation:

    def main(self):

        config_manager = ConfigurationManager()

        model_config = config_manager.get_model_inference_config()
        faiss_config = config_manager.get_faiss_config()
        pca_config = config_manager.get_pca_config()
        validation_config = config_manager.get_data_validation_config()

        test_dir = (
            validation_config.dataset_dir
            / "classification_task"
            / "test"
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

        evaluator = RetrievalEvaluator(
            query_feature_extractor=query_feature_extractor,
            query_pca=query_pca,
            similarity_search=similarity_search,
            model_config=model_config,
            class_names=model_config.class_names,
            test_dir=test_dir,
            samples_per_class=20,
            top_k=faiss_config.top_k,
            random_seed=42
        )

        results = evaluator.evaluate()

        evaluator.summarize(results)


if __name__ == "__main__":

    try:
        logger.info(
            ">>>>>> Stage 09 Retrieval Evaluation Started <<<<<<"
        )

        stage = Stage09RetrievalEvaluation()
        stage.main()

        logger.info(
            ">>>>>> Stage 09 Retrieval Evaluation Completed <<<<<<"
        )

    except Exception as e:
        logger.exception(e)
        raise CustomException(e, sys)
