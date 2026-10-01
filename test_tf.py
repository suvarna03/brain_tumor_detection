import pickle

import faiss
import numpy as np


index_file = "artifacts/faiss_index/faiss.index"
metadata_file = "artifacts/faiss_index/metadata.pkl"


# Load FAISS index
index = faiss.read_index(index_file)


# Load metadata
with open(
    metadata_file,
    "rb"
) as file:

    metadata = pickle.load(file)


labels = metadata["labels"]
image_paths = metadata["image_paths"]


print("FAISS index dimension:", index.d)
print("FAISS index size:", index.ntotal)

print("Labels shape:", labels.shape)
print("Image paths shape:", image_paths.shape)

print(
    "NaN labels:",
    np.isnan(labels).sum()
)

print(
    "Number of image paths:",
    len(image_paths)
)

print(
    "First label:",
    labels[0]
)

print(
    "First image path:",
    image_paths[0]
)