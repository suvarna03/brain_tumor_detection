import pickle
from pathlib import Path

import faiss
import numpy as np


class SimilaritySearch:

    def __init__(self, config):

        self.config = config

    def load_reduced_features(self):

        with open(self.config.reduced_feature_file,"rb") as file:

            data = pickle.load(file)

        features = data["features"]
        labels = data["labels"]
        image_paths = data["image_paths"]

        return features, labels, image_paths

    def build_index(self):

        features, labels, image_paths = (self.load_reduced_features())

        print("Reduced feature shape:",features.shape)

        print("Labels shape:",labels.shape)

        print("Image paths count:",len(image_paths))

        features = np.asarray(features,dtype=np.float32)

        faiss.normalize_L2(features)

        dimension = features.shape[1]
        index = faiss.IndexFlatIP(dimension)

        index.add(features)

        print("FAISS index dimension:",index.d)

        print("FAISS index size:",
            index.ntotal)
        
        self.config.root_dir.mkdir(parents=True,exist_ok=True)

        faiss.write_index(index,str(self.config.index_file))

        metadata = {"labels": labels,
                    "image_paths": image_paths}

        with open(self.config.metadata_file,"wb") as file:

            pickle.dump(metadata,file)

        print("FAISS index saved to:",self.config.index_file)

        print(
            "Metadata saved to:",self.config.metadata_file)

        return index

    def search(self, query_embedding, top_k=None):

        index = faiss.read_index(
            str(self.config.index_file))

        with open(
            self.config.metadata_file,"rb") as file:

            metadata = pickle.load(file)

        labels = metadata["labels"]
        image_paths = metadata["image_paths"]

        if top_k is None:
            top_k = self.config.top_k

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32)

        # Make sure query has shape (1, 256)
        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(
                query_embedding,axis=0)

        faiss.normalize_L2(query_embedding)

        similarity_scores, indices = index.search(
            query_embedding,top_k)

        results = []

        for rank, (score, index_id) in enumerate(
            zip(similarity_scores[0],
                indices[0]),
            start=1):

            result = {
                "rank": rank,
                "index": int(index_id),
                "similarity_score": float(score),
                "label": int(labels[index_id]),
                "image_path": str(image_paths[index_id])}

            results.append(result)

        return results