import pickle

import numpy as np


class QueryPCA:

    def __init__(self, pca_path):

        self.pca_path = pca_path

        with open(self.pca_path,"rb") as file:

            self.pca = pickle.load(file)

    def transform(self, features):

        features = np.asarray(features,
            dtype=np.float32)

        reduced_features = self.pca.transform(features)

        return reduced_features