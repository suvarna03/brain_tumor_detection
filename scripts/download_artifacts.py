
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

repo_id = os.getenv("HF_REPO_ID")
token = os.getenv("HF_TOKEN")

if not repo_id or not token:
    raise ValueError("HF_REPO_ID and HF_TOKEN must be set.")

artifacts = {
    "resnet50_finetuned.keras": "artifacts/model_trainer/resnet50_finetuned.keras",
    "pca.pkl": "artifacts/pca/pca.pkl",
    "faiss.index": "artifacts/faiss_index/faiss.index",
    "metadata.pkl": "artifacts/faiss_index/metadata.pkl",
}

for filename, destination in artifacts.items():
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token=token,
    )

    destination_path.write_bytes(Path(downloaded_path).read_bytes())
    print(f"Downloaded: {destination}")
