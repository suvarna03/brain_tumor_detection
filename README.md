# Brain Tumor Classification and Similar MRI Retrieval

A deep learning computer vision project that classifies brain MRI images into four categories, generates Grad-CAM visual explanations, and retrieves visually similar MRI images using deep feature embeddings.

## Project Overview

This project combines a convolutional neural network (CNN), visual explainability, and similarity-based image retrieval in a single application.

Given an MRI image, the system:

- Predicts one of four classes: glioma, meningioma, no tumor, or pituitary tumor.
- Returns the predicted class, confidence score, and class probabilities.
- Generates a Grad-CAM heatmap to visualize regions contributing to the prediction.
- Retrieves similar MRI images using deep feature embeddings, PCA, and FAISS.
- Displays results through a Streamlit application connected to a FastAPI backend.

**Important:** This is a research and educational prototype, not a clinically validated medical diagnostic system.

## Key Features

- Four-class brain MRI classification using ResNet50.
- Transfer learning and fine-tuning with TensorFlow/Keras.
- Grad-CAM model explainability.
- Deep feature extraction using 2,048-dimensional embeddings.
- PCA dimensionality reduction to 256 features.
- Similar-image retrieval using FAISS.
- REST API built with FastAPI.
- Interactive application built with Streamlit.
- Modular pipeline architecture and YAML-based configuration.

## Technology Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Deep learning | TensorFlow, Keras |
| Image processing | Pillow, NumPy |
| Dimensionality reduction | PCA, scikit-learn |
| Similarity retrieval | FAISS |
| Backend API | FastAPI, Uvicorn |
| Frontend | Streamlit |
| Configuration | YAML |

## Dataset

This project uses the BRISC 2025 brain MRI dataset.

**Dataset:** [BRISC 2025 — Kaggle](https://www.kaggle.com/datasets/briscdataset/brisc2025)

The classification task contains four classes:

| Class | Description |
|---|---|
| `glioma` | Glioma |
| `meningioma` | Meningioma |
| `no_tumor` | No tumor |
| `pituitary` | Pituitary tumor |

The dataset also includes a segmentation task with MRI images and masks. The classification and similarity-retrieval workflow described here uses the classification images.

Download the dataset from Kaggle and place it in the location expected by `config.yaml`.

## Model Architecture

The classification model uses a fine-tuned ResNet50 backbone with a custom classification head.

The architecture includes:

1. ResNet50 convolutional feature extractor.
2. Global Average Pooling.
3. Batch Normalization.
4. Dense layer with 256 units and ReLU activation.
5. Dropout.
6. Four-unit softmax output layer.

The model accepts RGB images resized to 224 × 224 pixels. Preprocessing is embedded in the saved model; therefore, additional image normalization must not be applied before inference.

## Model Evaluation

The selected fine-tuned model achieved the following results on the evaluated test set.

| Metric | Result |
|---|---:|
| Accuracy | 94.6% |
| Macro F1-score | 0.9496 |
| Test-set size | 1,000 images |

### Confusion Matrix

Class order: `glioma`, `meningioma`, `no_tumor`, `pituitary`.

| Actual / Predicted | Glioma | Meningioma | No tumor | Pituitary |
|---|---:|---:|---:|---:|
| Glioma | 229 | 22 | 0 | 3 |
| Meningioma | 10 | 282 | 6 | 8 |
| No tumor | 0 | 0 | 140 | 0 |
| Pituitary | 1 | 4 | 0 | 295 |

These metrics represent performance on the evaluated dataset and do not establish clinical effectiveness or generalization to independent clinical populations.

## Similar MRI Retrieval

The retrieval pipeline finds images with similar deep feature representations.

### Workflow

1. Extract image embeddings using the trained ResNet50 model.
2. Reduce embedding dimensions from 2,048 to 256 using PCA.
3. Build a FAISS similarity index from the reduced embeddings.
4. Extract features from the query image.
5. Apply the fitted PCA transformation.
6. Search the FAISS index for the top-k similar images.
7. Return image metadata and similarity scores.

The existing index contains 5,000 image embeddings.

Similarity scores measure similarity in the learned feature space. They are not class probabilities or diagnostic confidence scores.

## Grad-CAM Explainability

Grad-CAM generates a heatmap using gradients from the target convolutional layer `conv5_block3_3_conv`.

The heatmap is resized and overlaid on the original MRI image to support visual inspection of the regions influencing the model's selected class score.

Grad-CAM provides a model explanation visualization, not proof that a prediction is medically correct.

## Project Structure

The following tree reflects the source and root-level files identified during project inspection. Artifact and documentation subdirectories are shown at a high level.

```text
brain_tumor_detection/
├── app.py
├── config.yaml
├── setup.py
├── requirements.txt
├── README.md
├── .gitignore
├── diagnose_models.py
├── inspect_features.py
├── inspect_retrieval_artifacts.py
├── test.py
├── config/
├── docs/
├── research/
├── artifacts/
├── logs/
└── src/
    └── brain_tumor_detection/
        ├── api/
        │   └── main.py
        ├── components/
        │   ├── data_ingestion.py
        │   ├── data_validation.py
        │   ├── feature_extractor.py
        │   ├── gradcam.py
        │   ├── model_inference.py
        │   ├── model_trainer.py
        │   ├── pca_transformer.py
        │   ├── query_feature_extractor.py
        │   ├── query_pca.py
        │   ├── report_generator.py
        │   ├── retrieval_evaluator.py
        │   └── similarity_search.py
        ├── config/
        ├── constants/
        ├── entity/
        ├── exception/
        ├── logger/
        ├── pipelines/
        │   ├── stage_01_data_ingestion.py
        │   ├── stage_02_data_validation.py
        │   ├── stage_03_model_training.py
        │   ├── stage_04_model_inference.py
        │   ├── stage_05_feature_extraction.py
        │   ├── stage_06_pca.py
        │   ├── stage_07_similarity_search.py
        │   ├── stage_08_query_similarity.py
        │   ├── stage_09_retrieval_evaluation.py
        │   ├── stage_10_combined_inference.py
        │   ├── stage_11_candidate_evaluation.py
        │   ├── test_gradcam.py
        │   └── test_service.py
        ├── services/
        │   └── inference_service.py
        └── utils/
            └── common.py
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
cd brain_tumor_detection
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset and model artifacts

Ensure the dataset and required inference artifacts exist at the paths specified in `config.yaml`.

The inference application requires the trained model, PCA transformation, FAISS index, and retrieval metadata. These artifacts must be generated by the appropriate pipeline stages or obtained from the project's published artifact location.

## Running the Application

### Start the FastAPI backend

Run from the project root:

```bash
uvicorn src.brain_tumor_detection.api.main:app --reload
```

API documentation is available at:

- Interactive documentation: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

### Start the Streamlit application

Open a second terminal, activate the virtual environment, and run:

```bash
streamlit run app.py
```

Upload a supported MRI image and run the analysis to view the classification result, confidence, Grad-CAM overlay, and similar MRI images.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API information |
| GET | `/health` | Service health check |
| POST | `/predict` | Classification, Grad-CAM explanation, and similar-image retrieval |
| GET | `/docs` | Interactive API documentation |

The `/predict` endpoint accepts an uploaded JPG, JPEG, or PNG image.

## Pipeline Stages

The project includes pipeline scripts for:

1. Data ingestion
2. Data validation
3. Model training
4. Model inference
5. Feature extraction
6. PCA dimensionality reduction
7. Similarity-index creation
8. Query similarity search
9. Retrieval evaluation
10. Combined inference
11. Candidate-model evaluation

The Streamlit application uses the existing inference artifacts; it does not retrain the model for each prediction.

## Limitations

- Classification predictions may be incorrect.
- Grad-CAM visualizations do not establish prediction correctness.
- Similar-image retrieval does not confirm a diagnosis.
- Results on the evaluated dataset may not generalize to independent clinical populations.
- This application is not intended for clinical diagnosis or medical decision-making.

## Future Improvements

- Automated testing for API inputs and inference behavior.
- Model and artifact version tracking.
- Experiment tracking and reproducible evaluation.
- Additional retrieval-specific evaluation metrics.
- Testing on independent datasets.
- Deployment monitoring and security hardening.