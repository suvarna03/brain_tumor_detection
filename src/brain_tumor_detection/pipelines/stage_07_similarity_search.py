from brain_tumor_detection.config.configuration import (
    ConfigurationManager
)

from brain_tumor_detection.components.similarity_search import (
    SimilaritySearch
)


class Stage07SimilaritySearch:

    def main(self):

        print("Stage 07 FAISS Similarity Search Started")

        config_manager = ConfigurationManager()

        faiss_config = (
            config_manager.get_faiss_config()
        )

        similarity_search = SimilaritySearch(
            config=faiss_config
        )

        similarity_search.build_index()

        print("Stage 07 FAISS Similarity Search Completed")


if __name__ == "__main__":

    stage = Stage07SimilaritySearch()

    stage.main()