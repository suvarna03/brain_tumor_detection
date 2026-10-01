import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input


class QueryFeatureExtractor:

    def __init__(self, config):

        self.config = config

        self.model = load_model(self.config.model_path,
            custom_objects={
                "preprocess_input": preprocess_input},
            compile=False)

        # Run one dummy input so all model layers are built
        dummy_input = tf.zeros(
            shape=(1,
                self.config.image_size[0],
                self.config.image_size[1],
                3),
            dtype=tf.float32)

        self.model(dummy_input,
            training=False)

        # Extract output from the GAP layer
        feature_layer = self.model.get_layer("global_average_pooling2d")

        self.feature_model = tf.keras.Model(
            inputs=self.model.inputs,
            outputs=feature_layer.output)

    def extract(self, image_array):

        image_array = np.asarray(image_array,
            dtype=np.float32)

        if image_array.ndim == 3:
            image_array = np.expand_dims(image_array,
                axis=0)

        features = self.feature_model.predict(image_array,
            verbose=0)

        return features