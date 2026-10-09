from io import BytesIO
from pathlib import Path
import uuid

import numpy as np
from PIL import Image

from brain_tumor_detection.components.gradcam import GradCAM
from brain_tumor_detection.components.model_inference import ModelInference
from brain_tumor_detection.components.query_feature_extractor import QueryFeatureExtractor
from brain_tumor_detection.components.query_pca import QueryPCA
from brain_tumor_detection.components.similarity_search import SimilaritySearch


class InferenceService:

    def __init__(
        self,
        model_config,
        faiss_config,
        pca_path,
        gradcam_output_dir="artifacts/gradcam"
    ):
        self.model_config = model_config
        self.faiss_config = faiss_config
        self.pca_path = Path(pca_path)

        self.gradcam_output_dir = Path(gradcam_output_dir)
        self.gradcam_output_dir.mkdir(parents=True, exist_ok=True)

        self.model_inference = ModelInference(model_config)
        self.query_feature_extractor = QueryFeatureExtractor(model_config)
        self.query_pca = QueryPCA(self.pca_path)
        self.similarity_search = SimilaritySearch(faiss_config)

        self.gradcam = GradCAM(
            model_path=model_config.model_path,
            target_layer="conv5_block3_3_conv"
        )

    def preprocess_image(self, image_bytes):

        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        original_image = np.asarray(
            image,
            dtype=np.uint8
        )

        image = image.resize(
            (
                self.model_config.image_size[0],
                self.model_config.image_size[1]
            )
        )

        image_array = np.asarray(
            image,
            dtype=np.float32
        )

        image_array = np.expand_dims(image_array, axis=0)

        return image_array, original_image

    def predict(self, image_bytes):

        image_array, original_image = self.preprocess_image(image_bytes)

        classification_result = self.model_inference.predict(image_array)

        predicted_class = classification_result["predicted_class"]
        predicted_class_id = self.model_config.class_names.index(
            predicted_class
        )

        heatmap = self.gradcam.generate_heatmap(
            image_array=image_array,
            class_index=predicted_class_id
        )

        overlay = self.gradcam.create_overlay(
            original_image=original_image,
            heatmap=heatmap
        )

        filename = f"gradcam_{uuid.uuid4().hex}.jpg"
        gradcam_path = self.gradcam_output_dir / filename

        Image.fromarray(overlay).save(gradcam_path)

        features = self.query_feature_extractor.extract(image_array)
        reduced_features = self.query_pca.transform(features)

        similarity_results = self.similarity_search.search(
            reduced_features,
            top_k=self.faiss_config.top_k
        )

        for result in similarity_results:

            label_id = result["label"]
            result["label"] = self.model_config.class_names[label_id]

            result.pop("index", None)

            result["image_name"] = Path(
                result["image_path"]
            ).name

            result.pop("image_path", None)

        return {
            "classification": classification_result,
            "explanation": {
                "method": "Grad-CAM",
                "target_layer": "conv5_block3_3_conv",
                "image_url": f"/gradcam/{filename}"
            },
            "similar_images": similarity_results
        }