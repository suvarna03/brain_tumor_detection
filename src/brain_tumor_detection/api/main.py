from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from brain_tumor_detection.config.configuration import ConfigurationManager
from brain_tumor_detection.services.inference_service import InferenceService


app = FastAPI(
    title="Brain Tumor Detection API",
    description="MRI classification and similarity search API",
    version="1.0.0"
)

app.mount(
    "/gradcam",
    StaticFiles(directory="artifacts/gradcam"),
    name="gradcam"
)

app.mount(
    "/reference-images",
    StaticFiles(
        directory=(
            "artifacts/data_ingestion/"
            "brisc2025/brisc2025/"
            "classification_task/train"
        )
    ),
    name="reference-images"
)


config_manager = ConfigurationManager()

model_config = config_manager.get_model_inference_config()
faiss_config = config_manager.get_faiss_config()

pca_path = Path("artifacts/pca/pca.pkl")

inference_service = InferenceService(
    model_config=model_config,
    faiss_config=faiss_config,
    pca_path=pca_path
)


@app.get("/")
def root():
    return {
        "message": "Brain Tumor Detection API",
        "version": "1.0.0"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "brain-tumor-detection-api"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    allowed_extensions = {".jpg", ".jpeg", ".png"}
    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image format. Use JPG, JPEG, or PNG."
        )

    try:
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty."
            )

        return inference_service.predict(image_bytes)

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(error)}"
        )