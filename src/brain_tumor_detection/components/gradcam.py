from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model


class GradCAM:

    def __init__(
        self,
        model_path,
        target_layer="conv5_block3_3_conv"
    ):

        self.model_path = Path(model_path)
        self.target_layer_name = target_layer

        self.model = load_model(
            self.model_path,
            custom_objects={
                "preprocess_input": preprocess_input
            },
            compile=False
        )

        self.preprocess_layer = self.model.get_layer("lambda")
        self.base_model = self.model.get_layer("resnet50")

        self.target_conv_layer = self.base_model.get_layer(
            self.target_layer_name
        )

        self.base_grad_model = tf.keras.Model(
            inputs=self.base_model.input,
            outputs=[
                self.target_conv_layer.output,
                self.base_model.output
            ]
        )

        self.gap_layer = self.model.get_layer(
            "global_average_pooling2d"
        )
        self.bn_layer = self.model.get_layer(
            "batch_normalization"
        )
        self.dense_layer = self.model.get_layer("dense")
        self.dropout_layer = self.model.get_layer("dropout")
        self.classifier_layer = self.model.get_layer("dense_1")

    def generate_heatmap(self, image_array, class_index):

        image_array = np.asarray(
            image_array,
            dtype=np.float32
        )

        if image_array.ndim == 3:
            image_array = np.expand_dims(
                image_array,
                axis=0
            )

        image_tensor = tf.convert_to_tensor(
            image_array,
            dtype=tf.float32
        )

        with tf.GradientTape() as tape:

            preprocessed_image = self.preprocess_layer(
                image_tensor
            )

            conv_outputs, base_output = self.base_grad_model(
                preprocessed_image,
                training=False
            )

            x = self.gap_layer(base_output)
            x = self.bn_layer(x, training=False)
            x = self.dense_layer(x, training=False)
            x = self.dropout_layer(x, training=False)

            predictions = self.classifier_layer(
                x,
                training=False
            )

            class_output = predictions[:, class_index]

        gradients = tape.gradient(
            class_output,
            conv_outputs
        )

        if gradients is None:
            raise RuntimeError(
                "Gradients could not be calculated "
                "for the target convolution layer."
            )

        pooled_gradients = tf.reduce_mean(
            gradients,
            axis=(1, 2)
        )

        conv_outputs = conv_outputs[0]
        pooled_gradients = pooled_gradients[0]

        weighted_features = (
            conv_outputs * pooled_gradients
        )

        heatmap = tf.reduce_mean(
            weighted_features,
            axis=-1
        )

        heatmap = tf.maximum(
            heatmap,
            0
        )

        max_value = tf.reduce_max(heatmap)

        heatmap = heatmap / (
            max_value + 1e-8
        )

        return heatmap.numpy()

    def create_overlay(
        self,
        original_image,
        heatmap,
        alpha=0.4
    ):

        original_image = np.asarray(
            original_image,
            dtype=np.uint8
        )

        height, width = original_image.shape[:2]

        heatmap_image = Image.fromarray(
            np.uint8(heatmap * 255)
        )

        heatmap_image = heatmap_image.resize(
            (width, height),
            Image.Resampling.BILINEAR
        )

        heatmap = (
            np.asarray(
                heatmap_image,
                dtype=np.float32
            ) / 255.0
        )

        red = np.clip(
            2 * heatmap,
            0,
            1
        )

        green = np.clip(
            2 * (1 - np.abs(heatmap - 0.5)),
            0,
            1
        )

        blue = np.clip(
            2 * (1 - heatmap),
            0,
            1
        )

        heatmap_rgb = np.stack(
            [red, green, blue],
            axis=-1
        )

        heatmap_rgb = (
            heatmap_rgb * 255
        ).astype(np.uint8)

        overlay = (
            (1 - alpha)
            * original_image.astype(np.float32)
            + alpha
            * heatmap_rgb.astype(np.float32)
        )

        return np.clip(
            overlay,
            0,
            255
        ).astype(np.uint8)