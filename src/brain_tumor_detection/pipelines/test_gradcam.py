from pathlib import Path

import numpy as np
from PIL import Image

from brain_tumor_detection.components.gradcam import GradCAM
from brain_tumor_detection.config.configuration import ConfigurationManager


CLASS_NAMES = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary"
]


def main():

    config_manager = ConfigurationManager()

    model_config = (
        config_manager.get_model_inference_config()
    )

    image_path = Path(
        "artifacts/data_ingestion/"
        "brisc2025/brisc2025/"
        "classification_task/test/"
        "glioma/"
        "brisc2025_test_00001_gl_ax_t1.jpg"
    )

    print("Loading image...")

    image = Image.open(
        image_path
    ).convert("RGB")

    original_image = np.asarray(
        image,
        dtype=np.uint8
    )

    resized_image = image.resize(
        (
            model_config.image_size[0],
            model_config.image_size[1]
        )
    )

    image_array = np.asarray(
        resized_image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    print(
        "Image array shape:",
        image_array.shape
    )

    gradcam = GradCAM(
        model_path=model_config.model_path,
        target_layer="conv5_block3_3_conv"
    )

    print("Generating prediction...")

    predictions = gradcam.model.predict(
        image_array,
        verbose=0
    )

    predicted_class_id = int(
        np.argmax(predictions[0])
    )

    predicted_class = CLASS_NAMES[
        predicted_class_id
    ]

    confidence = float(
        predictions[0][predicted_class_id]
    )

    print(
        "Predicted class:",
        predicted_class
    )

    print(
        "Confidence:",
        f"{confidence:.4f}"
    )

    print(
        "Generating Grad-CAM heatmap..."
    )

    heatmap = gradcam.generate_heatmap(
        image_array=image_array,
        class_index=predicted_class_id
    )

    overlay = gradcam.create_overlay(
        original_image=original_image,
        heatmap=heatmap
    )

    output_dir = Path(
        "artifacts/gradcam"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir /
        "gradcam_test.jpg"
    )

    Image.fromarray(
        overlay
    ).save(
        output_path
    )

    print(
        "Grad-CAM image saved to:",
        output_path
    )


if __name__ == "__main__":
    main()