from pathlib import Path

from brain_tumor_detection.config.configuration import ConfigurationManager
from brain_tumor_detection.services.inference_service import InferenceService


def main():

    config_manager = ConfigurationManager()

    model_config = config_manager.get_model_inference_config()
    faiss_config = config_manager.get_faiss_config()

    pca_path = Path(
        "artifacts/pca/pca.pkl"
    )

    service = InferenceService(
        model_config=model_config,
        faiss_config=faiss_config,
        pca_path=pca_path
    )

    image_path = Path(
        "artifacts/data_ingestion/brisc2025/brisc2025/"
        "classification_task/test/glioma/"
        "brisc2025_test_00001_gl_ax_t1.jpg"
    )

    with open(image_path, "rb") as file:
        image_bytes = file.read()

    result = service.predict(image_bytes)

    print("\nCLASSIFICATION")
    print(result["classification"])

    print("\nSIMILAR IMAGES")

    for image in result["similar_images"]:
        print(image)


if __name__ == "__main__":
    main()