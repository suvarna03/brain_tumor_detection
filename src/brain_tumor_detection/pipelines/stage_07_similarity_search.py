
import sys

from brain_tumor_detection.components.similarity_search import (
    SimilaritySearch
)
from brain_tumor_detection.config.configuration import ConfigurationManager
from brain_tumor_detection.exception import CustomException
from brain_tumor_detection.logger import logger


class Stage07SimilaritySearch:

    def main(self):

        config_manager = ConfigurationManager()

        faiss_config = config_manager.get_faiss_config()

        similarity_search = SimilaritySearch(
            config=faiss_config
        )

        similarity_search.build_index()


if __name__ == "__main__":

    try:
        logger.info(
            ">>>>>> Stage 07 FAISS Similarity Search Started <<<<<<"
        )

        stage = Stage07SimilaritySearch()
        stage.main()

        logger.info(
            ">>>>>> Stage 07 FAISS Similarity Search Completed <<<<<<"
        )

    except Exception as e:
        logger.exception(e)
        raise CustomException(e, sys)
