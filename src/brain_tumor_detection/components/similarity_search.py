import pickle

import faiss
import numpy as np

class SimilaritySearch:

    def __init__(self, config):

        self.config = config

    def load_reduced_features(self):

        with open(
            self.config.reduced_feature_file,
            "rb") as file:
            data = pickle.load(file)

        return (
            data["features"],
            data["labels"],
            data["image_paths"])

    def build_index(self):

        features, labels, image_paths = (
            self.load_reduced_features())

        print("Reduced feature shape:", features.shape)
        print("Labels shape:", labels.shape)
        print("Image paths count:", len(image_paths))

        features = np.asarray(features,dtype=np.float32)

        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError("Features must be a non-empty 2D array.")

        if not np.isfinite(features).all():
            raise ValueError("Features contain invalid values.")

        if np.any(np.linalg.norm(features, axis=1) == 0):
            raise ValueError("Features contain zero-length vectors.")

        if (len(labels) != len(features)
            or len(image_paths) != len(features)):
            raise ValueError(
                "Features, labels, and image paths "
                "must have equal lengths.")

        features = features.copy()
        faiss.normalize_L2(features)

        dimension = features.shape[1]

        index = faiss.IndexFlatIP(dimension)
        index.add(features)

        print("FAISS index dimension:", index.d)
        print("FAISS index size:", index.ntotal)

        self.config.root_dir.mkdir(
            parents=True,
            exist_ok=True)

        self.config.index_file.parent.mkdir(
            parents=True,
            exist_ok=True)

        self.config.metadata_file.parent.mkdir(
            parents=True,
            exist_ok=True)

        faiss.write_index(index,str(self.config.index_file))

        metadata = {
            "labels": np.asarray(labels),
            "image_paths": np.asarray(image_paths)}

        with open(
            self.config.metadata_file,
            "wb") as file:
            pickle.dump(metadata, file)

        print(
            "FAISS index saved to:",
            self.config.index_file
        )
        print(
            "Metadata saved to:",
            self.config.metadata_file)

        return index

    def search(self, query_embedding, top_k=None):

        index = faiss.read_index(str(self.config.index_file))

        with open(
            self.config.metadata_file,
            "rb") as file:
            metadata = pickle.load(file)

        labels = metadata["labels"]
        image_paths = metadata["image_paths"]

        if (len(labels) != index.ntotal
            or len(image_paths) != index.ntotal):
            raise ValueError("FAISS index and metadata sizes do not match." )

        if top_k is None:
            top_k = self.config.top_k

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        if index.ntotal == 0:
            raise ValueError("FAISS index is empty.")

        top_k = min(top_k, index.ntotal)

        query_embedding = np.asarray(query_embedding,dtype=np.float32)

        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding,axis=0)

        if (query_embedding.ndim != 2
            or query_embedding.shape[0] == 0):
            raise ValueError(
                "Query embedding must be a non-empty "
                "1D or 2D array.")

        if query_embedding.shape[1] != index.d:
            raise ValueError(f"Expected embedding dimension {index.d}, "
                f"received {query_embedding.shape[1]}.")

        if not np.isfinite(query_embedding).all():
            raise ValueError("Query embedding contains invalid values.")

        if np.any(np.linalg.norm(query_embedding, axis=1) == 0):
            raise ValueError("Query embedding contains a zero-length vector.")

        query_embedding = query_embedding.copy()
        faiss.normalize_L2(query_embedding)

        similarity_scores, indices = index.search(query_embedding,top_k)

        results = []

        for rank, (score, index_id) in enumerate(
            zip(
                similarity_scores[0],
                indices[0]),start=1):

            if index_id < 0 or index_id >= len(labels):
                continue

            results.append({
                "rank": rank,
                "index": int(index_id),
                "similarity_score": float(score),
                "label": int(labels[index_id]),
                "image_path": str(image_paths[index_id])
            })

        return results
