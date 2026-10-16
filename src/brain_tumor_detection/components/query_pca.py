import pickle

import numpy as np


class QueryPCA:

    def __init__(self, pca_path):

        self.pca_path = pca_path

        with open(self.pca_path, "rb") as file:
            self.pca = pickle.load(file)

    def transform(self, features):

        features = np.asarray(
            features,
            dtype=np.float32
        )

        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError(
                "Features must be a non-empty 2D array."
            )

        if features.shape[1] != self.pca.n_features_in_:
            raise ValueError(
                f"Expected {self.pca.n_features_in_} input features, "
                f"received {features.shape[1]}."
            )

        if not np.isfinite(features).all():
            raise ValueError(
                "Features contain invalid values."
            )

        reduced_features = self.pca.transform(features)

        reduced_features = np.asarray(
            reduced_features,
            dtype=np.float32
        )

        if not np.isfinite(reduced_features).all():
            raise ValueError(
                "Reduced features contain invalid values."
            )

        return reduced_features