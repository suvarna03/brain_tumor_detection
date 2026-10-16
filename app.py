import os
import requests
import streamlit as st
from pathlib import Path

from brain_tumor_detection.components.report_generator import ReportGenerator


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
MAX_SIMILAR_IMAGES = 3

MAIN_IMAGE_WIDTH = 400
SIMILAR_IMAGE_WIDTH = 250

report_generator = ReportGenerator()


st.set_page_config(
    page_title="Brain Tumor Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(37, 99, 235, 0.18),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 90%,
                rgba(14, 165, 233, 0.12),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                #07152f 0%,
                #0b1f3f 50%,
                #07152f 100%
            );

        color: #f8fafc;
    }

    .main {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }

    .app-header {
        text-align: center;
        padding: 18px 0 24px 0;
    }

    .app-title {
        font-size: 2.25rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.5px;
        margin-bottom: 6px;
    }

    .app-subtitle {
        font-size: 0.95rem;
        color: #b8c7dc;
        font-weight: 500;
    }

    .section-heading {
        text-align: center;
        font-size: 1.35rem;
        font-weight: 750;
        color: #ffffff;
        margin-top: 8px;
        margin-bottom: 16px;
    }

    .upload-card {
        background: rgba(15, 35, 68, 0.90);
        border: 1px solid rgba(96, 165, 250, 0.30);
        border-radius: 16px;
        padding: 18px 22px 12px 22px;
        margin-bottom: 12px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.20);
    }

    .upload-title {
        text-align: center;
        color: #ffffff;
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .upload-description {
        text-align: center;
        color: #aebed4;
        font-size: 0.88rem;
    }

    [data-testid="stFileUploader"] {
        background: #ffffff !important;
        border: 2px solid #dbe4f0 !important;
        border-radius: 14px !important;
        padding: 12px !important;
        box-shadow: 0 8px 22px rgba(0, 0, 0, 0.15);
    }

    [data-testid="stFileUploader"] button {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border: 1px solid #2563eb !important;
        border-radius: 9px !important;
        font-weight: 700 !important;
        min-height: 42px !important;
        padding: 6px 20px !important;
        box-shadow: 0 4px 10px rgba(37, 99, 235, 0.25) !important;
    }

    [data-testid="stFileUploader"] button:hover {
        background-color: #eef4fb !important;
        border-color: #2563eb !important;
        color: #1d4ed8 !important;
    }

    [data-testid="stFileUploader"] button svg {
        color: #1d4ed8 !important;
        fill: #1d4ed8 !important;
    }

    [data-testid="stFileUploader"] section {
        background-color: #ffffff !important;
    }

    [data-testid="stFileUploader"] small {
        color: #52657d !important;
        font-weight: 500 !important;
    }

    [data-testid="stFileUploader"] label {
        color: #0b1f3f !important;
    }

    .upload-success {
        text-align: center;
        color: #86efac;
        font-size: 0.92rem;
        font-weight: 750;
        padding: 8px 0;
    }

    .image-section-card {
        background: rgba(15, 35, 68, 0.90);
        border: 1px solid rgba(96, 165, 250, 0.30);
        border-radius: 18px;
        padding: 16px 12px 14px 12px;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.22);
        min-height: 80px;
    }

    .image-title {
        text-align: center;
        color: #ffffff;
        font-size: 1.25rem;
        font-weight: 800;
        margin-bottom: 15px;
    }

    .image-caption {
        width: 100%;
        box-sizing: border-box;
        text-align: center;
        color: #ffffff;
        font-size: 0.90rem;
        font-weight: 750;
        padding: 10px 12px;
        margin-top: 12px;
        border-radius: 9px;
        background: rgba(37, 99, 235, 0.20);
        border: 1px solid rgba(96, 165, 250, 0.25);
        min-height: 42px;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .explanation-note {
        width: 100%;
        box-sizing: border-box;
        text-align: center;
        color: #ffffff;
        font-size: 0.88rem;
        font-weight: 750;
        line-height: 1.45;
        padding: 10px 14px;
        margin-top: 12px;
        border-radius: 9px;
        background: rgba(245, 158, 11, 0.18);
        border: 1px solid rgba(251, 191, 36, 0.35);
        min-height: 42px;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .result-card {
        background:
            linear-gradient(
                145deg,
                rgba(29, 78, 216, 0.40),
                rgba(15, 35, 68, 0.92)
            );

        border: 1px solid rgba(96, 165, 250, 0.30);
        border-radius: 16px;
        padding: 22px;
        text-align: center;
        min-height: 145px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.20);
    }

    .result-label {
        color: #b8c7dc;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 7px;
    }

    .result-class {
        color: #ffffff;
        font-size: 1.7rem;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .result-confidence {
        color: #ffffff;
        font-size: 1rem;
    }

    .similar-description {
        text-align: center;
        color: #aebed4;
        font-size: 0.85rem;
        margin-bottom: 16px;
    }

    .similar-info {
        width: 100%;
        box-sizing: border-box;
        background:
            linear-gradient(
                145deg,
                rgba(14, 116, 144, 0.35),
                rgba(15, 35, 68, 0.95)
            );

        border: 1px solid rgba(45, 212, 191, 0.35);
        border-radius: 10px;
        padding: 10px;
        margin-top: 8px;
        text-align: center;
        min-height: 60px;
    }

    .similar-label {
        color: #ffffff;
        font-size: 0.90rem;
        font-weight: 800;
    }

    .similar-score {
        color: #99f6e4;
        font-size: 0.80rem;
        font-weight: 650;
        margin-top: 4px;
    }

    div[data-testid="stDownloadButton"] > button {
        background: #38bdf8 !important;
        color: #082f49 !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 800 !important;
        font-size: 1rem !important;
        min-height: 46px !important;
        transition: all 0.2s ease-in-out !important;
    }

    div[data-testid="stDownloadButton"] > button:hover {
        background: #7dd3fc !important;
        color: #082f49 !important;
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(56, 189, 248, 0.3);
    }

    .research-note {
        text-align: center;
        color: #cbd5e1;
        font-size: 0.78rem;
        line-height: 1.5;
        padding: 12px;
    }

    .footer-note {
        text-align: center;
        color: #8193ad;
        font-size: 0.75rem;
        margin-top: 18px;
        padding-bottom: 10px;
    }

    .stButton > button {
        border-radius: 10px !important;
        font-weight: 700 !important;
        min-height: 44px !important;
    }

    hr {
        border-color: rgba(148, 163, 184, 0.18);
    }

    [data-testid="stImage"] {
        display: flex;
        justify-content: center;
    }

    [data-testid="stImage"] img {
        margin-left: auto;
        margin-right: auto;
    }

    p {
        color: #dbeafe;
    }

    </style>
    """,
    unsafe_allow_html=True
)


st.html(
    """
    <div class="app-header">
        <div class="app-title">
            🧠 Brain Tumor Analysis
        </div>

        <div class="app-subtitle">
            MRI Classification • Model Explainability • Similarity Search
        </div>
    </div>
    """
)

# Sample MRI downloads for visitors to test the demo
st.markdown("### 📥 Download Sample MRI Images")
st.write(
    "New to the demo? Download a sample MRI image below, "
    "then upload it to test the model."
)

sample_dir = Path(__file__).parent / "sample_images"
sample_images = [
    ("Glioma", "glioma.jpg"),
    ("Meningioma", "meningioma.jpg"),
    ("No Tumor", "no_tumor.jpg"),
    ("Pituitary", "pituitary.jpg"),
]
sample_columns = st.columns(4)

for column, (label, filename) in zip(sample_columns, sample_images):
    image_path = sample_dir / filename

    with column:
        if image_path.is_file():
            with open(image_path, "rb") as image_file:
                st.download_button(
                    label=f"Download {label}",
                    data=image_file.read(),
                    file_name=filename,
                    mime="image/jpeg",
                    use_container_width=True,
                )

try:
    health_response = requests.get(
        f"{API_URL}/health",
        timeout=5
    )
    api_available = health_response.status_code == 200

except requests.exceptions.RequestException:
    api_available = False


if not api_available:
    st.error("Brain Tumor Detection API is not running.")

    st.code(
        "uvicorn src.brain_tumor_detection.api.main:app --reload",
        language="bash"
    )

    st.stop()


st.html(
    """
    <div class="upload-card">
        <div class="upload-title">
            Upload MRI Scan
        </div>

        <div class="upload-description">
            Upload a JPG, JPEG or PNG brain MRI image for analysis.
        </div>
    </div>
    """
)


uploaded_file = st.file_uploader(
    "Choose MRI image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)


if uploaded_file is not None:

    st.html(
        f"""
        <div class="upload-success">
            ✓ {uploaded_file.name} uploaded successfully
        </div>
        """
    )

    analyze_clicked = st.button(
        "🔍 Analyze MRI",
        type="primary",
        use_container_width=True
    )

else:
    analyze_clicked = False


if analyze_clicked:

    with st.spinner(
        "Analyzing MRI using the trained ResNet50 model..."
    ):
        try:
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type
                )
            }

            response = requests.post(
                f"{API_URL}/predict",
                files=files,
                timeout=120
            )

            if response.status_code != 200:
                st.error(f"Analysis failed: {response.text}")
                st.stop()

            result = response.json()

        except requests.exceptions.RequestException as error:
            st.error(f"Unable to complete analysis: {error}")
            st.stop()


    classification = result["classification"]
    predicted_class = classification["predicted_class"]
    confidence = classification["confidence"]
    probabilities = classification["class_probabilities"]

    explanation = result["explanation"]
    similar_images = result["similar_images"][:MAX_SIMILAR_IMAGES]

    uploaded_image_bytes = uploaded_file.getvalue()
    gradcam_image_bytes = None
    similar_image_bytes = {}


    st.divider()

    st.html(
        """
        <div class="section-heading">
            MRI Preview & Model Explanation
        </div>
        """
    )


    original_col, explanation_col = st.columns(
        2,
        gap="large"
    )


    with original_col:

        st.html(
            """
            <div class="image-section-card">
                <div class="image-title">
                    Original MRI
                </div>
            </div>
            """
        )

        left_space, image_space, right_space = st.columns(
            [0.5, 2, 0.5]
        )

        with image_space:
            st.image(
                uploaded_file,
                width=MAIN_IMAGE_WIDTH
            )

        st.html(
            """
            <div class="image-caption">
                Uploaded MRI scan
            </div>
            """
        )


    with explanation_col:

        st.html(
            """
            <div class="image-section-card">
                <div class="image-title">
                    Model Explanation
                </div>
            </div>
            """
        )

        gradcam_url = (
            f"{API_URL}"
            f"{explanation['image_url']}"
        )

        try:
            gradcam_response = requests.get(
                gradcam_url,
                timeout=30
            )

            if gradcam_response.status_code == 200:

                gradcam_image_bytes = gradcam_response.content

                left_space, image_space, right_space = st.columns(
                    [0.5, 2, 0.5]
                )

                with image_space:
                    st.image(
                        gradcam_response.content,
                        width=MAIN_IMAGE_WIDTH
                    )

            else:
                st.warning("Grad-CAM visualization unavailable.")

        except requests.exceptions.RequestException:
            st.warning("Unable to load Grad-CAM visualization.")


        st.html(
            """
            <div class="explanation-note">
                Highlighted regions indicate areas that
                influenced the model's prediction.
            </div>
            """
        )


    st.divider()

    st.html(
        """
        <div class="section-heading">
            Analysis Result
        </div>
        """
    )


    result_col, probability_col = st.columns(
        [0.8, 1.6],
        gap="large"
    )


    with result_col:

        display_class = (
            predicted_class
            .replace("_", " ")
            .title()
        )

        st.html(
            f"""
            <div class="result-card">
                <div class="result-label">
                    Predicted MRI Type
                </div>

                <div class="result-class">
                    {display_class}
                </div>

                <div class="result-confidence">
                    Confidence:
                    <strong>
                        {confidence * 100:.2f}%
                    </strong>
                </div>
            </div>
            """
        )


    with probability_col:

        st.html(
            """
            <div class="section-heading">
                Probability by MRI Type
            </div>
            """
        )

        probability_data = {
            class_name.replace("_", " ").title(): probability
            for class_name, probability in probabilities.items()
        }

        st.bar_chart(
            probability_data,
            horizontal=True
        )


    st.divider()

    st.html(
        """
        <div class="section-heading">
            Similar MRI References
        </div>

        <div class="similar-description">
            Retrieved using ResNet50 embeddings, PCA
            and FAISS similarity search.
        </div>
        """
    )


    if similar_images:

        similar_columns = st.columns(
            len(similar_images),
            gap="large"
        )

        for column, image_result in zip(
            similar_columns,
            similar_images
        ):

            label = image_result["label"]
            similarity = image_result["similarity_score"]
            image_name = image_result["image_name"]

            image_url = (
                f"{API_URL}"
                f"/reference-images/"
                f"{label}/"
                f"{image_name}"
            )

            with column:

                try:
                    image_response = requests.get(
                        image_url,
                        timeout=30
                    )

                    if image_response.status_code == 200:

                        similar_image_bytes[
                            image_name
                        ] = image_response.content

                        left_space, image_space, right_space = st.columns(
                            [0.5, 2, 0.5]
                        )

                        with image_space:
                            st.image(
                                image_response.content,
                                width=SIMILAR_IMAGE_WIDTH
                            )

                    else:
                        st.warning("Reference unavailable.")

                except requests.exceptions.RequestException:
                    st.warning("Reference unavailable.")


                display_label = (
                    label
                    .replace("_", " ")
                    .title()
                )

                st.html(
                    f"""
                    <div class="similar-info">
                        <div class="similar-label">
                            {display_label}
                        </div>

                        <div class="similar-score">
                            Similarity: {similarity:.4f}
                        </div>
                    </div>
                    """
                )

    else:
        st.info("No similar MRI references were returned.")


    st.divider()


    if gradcam_image_bytes is not None:

        try:
            pdf_bytes = report_generator.generate_report(
                uploaded_image_bytes=uploaded_image_bytes,
                gradcam_image_bytes=gradcam_image_bytes,
                predicted_class=predicted_class,
                confidence=confidence,
                probabilities=probabilities,
                similar_images=similar_images,
                similar_image_bytes=similar_image_bytes,
            )

            st.download_button(
                label="📥 Download Analysis Report",
                data=pdf_bytes,
                file_name="brain_tumor_analysis_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        except Exception as error:
            st.error(
                f"Unable to generate analysis report: {error}"
            )

    else:
        st.warning(
            "Analysis report is unavailable because "
            "the Grad-CAM image could not be generated."
        )


    st.divider()

    st.html(
        """
        <div class="research-note">
            Research prototype only. This system is not intended
            for clinical diagnosis or treatment decisions.
        </div>
        """
    )


    st.html(
        """
        <div class="footer-note">
            ResNet50 • TensorFlow/Keras • Grad-CAM • PCA • FAISS
        </div>
        """
    )