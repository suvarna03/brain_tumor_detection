from datetime import datetime
from io import BytesIO
from pathlib import Path

from PIL import Image as PILImage

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    ParagraphStyle,
    getSampleStyleSheet
)
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle
)


class ReportGenerator:

    def __init__(self):

        self.page_width, self.page_height = A4

        styles = getSampleStyleSheet()

        self.title_style = ParagraphStyle(
            "ReportTitle",
            parent=styles["Title"],
            alignment=TA_CENTER,
            fontSize=20,
            leading=24,
            spaceAfter=8
        )

        self.subtitle_style = ParagraphStyle(
            "ReportSubtitle",
            parent=styles["Normal"],
            alignment=TA_CENTER,
            fontSize=10,
            textColor=colors.grey,
            spaceAfter=18
        )

        self.heading_style = ParagraphStyle(
            "ReportHeading",
            parent=styles["Heading2"],
            fontSize=14,
            leading=18,
            spaceBefore=12,
            spaceAfter=8
        )

        self.center_style = ParagraphStyle(
            "CenterText",
            parent=styles["Normal"],
            alignment=TA_CENTER,
            fontSize=9,
            leading=12
        )

    def prepare_image(
        self,
        image_bytes,
        max_width=3.2 * inch,
        max_height=3.2 * inch
    ):

        image = PILImage.open(
            BytesIO(image_bytes)
        )

        width, height = image.size

        scale = min(
            max_width / width,
            max_height / height
        )

        display_width = width * scale
        display_height = height * scale

        image_buffer = BytesIO()

        image.save(
            image_buffer,
            format="JPEG"
        )

        image_buffer.seek(0)

        return Image(
            image_buffer,
            width=display_width,
            height=display_height
        )

    def generate_report(
        self,
        uploaded_image_bytes,
        gradcam_image_bytes,
        predicted_class,
        confidence,
        probabilities,
        similar_images,
        similar_image_bytes,
        output_path=None
    ):

        pdf_buffer = BytesIO()

        document = SimpleDocTemplate(
            pdf_buffer,
            pagesize=A4,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40
        )

        story = []

        story.append(
            Paragraph(
                "Brain Tumor Analysis Report",
                self.title_style
            )
        )

        story.append(
            Paragraph(
                "MRI Classification • Grad-CAM • Similarity Search",
                self.subtitle_style
            )
        )

        timestamp = datetime.now().strftime(
            "%d %B %Y, %I:%M %p"
        )

        story.append(
            Paragraph(
                f"Generated: {timestamp}",
                self.center_style
            )
        )

        story.append(
            Spacer(1, 15)
        )

        story.append(
            Paragraph(
                "Analysis Result",
                self.heading_style
            )
        )

        display_class = (
            predicted_class
            .replace("_", " ")
            .title()
        )

        result_data = [
            [
                Paragraph(
                    "<b>Predicted MRI Type</b>",
                    self.center_style
                ),
                Paragraph(
                    "<b>Confidence</b>",
                    self.center_style
                )
            ],
            [
                Paragraph(
                    display_class,
                    self.center_style
                ),
                Paragraph(
                    f"{confidence * 100:.2f}%",
                    self.center_style
                )
            ]
        ]

        result_table = Table(
            result_data,
            colWidths=[
                3.2 * inch,
                3.2 * inch
            ]
        )

        result_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor("#dbeafe")
                    ),
                    (
                        "BACKGROUND",
                        (0, 1),
                        (-1, 1),
                        colors.HexColor("#eff6ff")
                    ),
                    (
                        "BOX",
                        (0, 0),
                        (-1, -1),
                        1,
                        colors.HexColor("#93c5fd")
                    ),
                    (
                        "INNERGRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.HexColor("#bfdbfe")
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "MIDDLE"
                    ),
                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        8
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        8
                    )
                ]
            )
        )

        story.append(result_table)
        story.append(Spacer(1, 15))

        story.append(
            Paragraph(
                "Class Probabilities",
                self.heading_style
            )
        )

        probability_data = [
            [
                Paragraph(
                    "<b>MRI Type</b>",
                    self.center_style
                ),
                Paragraph(
                    "<b>Probability</b>",
                    self.center_style
                )
            ]
        ]

        for class_name, probability in probabilities.items():

            display_name = (
                class_name
                .replace("_", " ")
                .title()
            )

            probability_data.append(
                [
                    Paragraph(
                        display_name,
                        self.center_style
                    ),
                    Paragraph(
                        f"{probability * 100:.2f}%",
                        self.center_style
                    )
                ]
            )

        probability_table = Table(
            probability_data,
            colWidths=[
                4.5 * inch,
                1.9 * inch
            ]
        )

        probability_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor("#dbeafe")
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.HexColor("#cbd5e1")
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "MIDDLE"
                    ),
                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        7
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        7
                    )
                ]
            )
        )

        story.append(probability_table)
        story.append(Spacer(1, 15))

        story.append(
            Paragraph(
                "MRI Visualization",
                self.heading_style
            )
        )

        original_image = self.prepare_image(
            uploaded_image_bytes
        )

        gradcam_image = self.prepare_image(
            gradcam_image_bytes
        )

        image_table = Table(
            [
                [
                    original_image,
                    gradcam_image
                ],
                [
                    Paragraph(
                        "<b>Original MRI</b>",
                        self.center_style
                    ),
                    Paragraph(
                        "<b>Grad-CAM Explanation</b>",
                        self.center_style
                    )
                ]
            ],
            colWidths=[
                3.2 * inch,
                3.2 * inch
            ]
        )

        image_table.setStyle(
            TableStyle(
                [
                    (
                        "ALIGN",
                        (0, 0),
                        (-1, -1),
                        "CENTER"
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "MIDDLE"
                    ),
                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        6
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        6
                    )
                ]
            )
        )

        story.append(image_table)
        story.append(Spacer(1, 8))

        story.append(
            Paragraph(
                "Highlighted regions indicate areas that "
                "influenced the model's prediction.",
                self.center_style
            )
        )

        story.append(Spacer(1, 15))

        story.append(
            Paragraph(
                "Similar MRI References",
                self.heading_style
            )
        )

        story.append(
            Paragraph(
                "Reference images retrieved using "
                "ResNet50 embeddings, PCA and FAISS "
                "similarity search.",
                self.center_style
            )
        )

        story.append(Spacer(1, 10))

        image_row = []
        info_row = []

        for image_result in similar_images:

            image_name = image_result["image_name"]
            label = image_result["label"]
            similarity = image_result["similarity_score"]

            image_bytes = similar_image_bytes.get(
                image_name
            )

            if image_bytes:

                reference_image = self.prepare_image(
                    image_bytes,
                    max_width=1.9 * inch,
                    max_height=1.9 * inch
                )

                image_row.append(reference_image)

                display_label = (
                    label
                    .replace("_", " ")
                    .title()
                )

                info_row.append(
                    Paragraph(
                        (
                            f"<b>{display_label}</b><br/>"
                            f"Similarity: {similarity:.4f}"
                        ),
                        self.center_style
                    )
                )

        if image_row:

            similar_table = Table(
                [
                    image_row,
                    info_row
                ],
                colWidths=[
                    2.1 * inch
                    for _ in image_row
                ]
            )

            similar_table.setStyle(
                TableStyle(
                    [
                        (
                            "ALIGN",
                            (0, 0),
                            (-1, -1),
                            "CENTER"
                        ),
                        (
                            "VALIGN",
                            (0, 0),
                            (-1, -1),
                            "MIDDLE"
                        ),
                        (
                            "TOPPADDING",
                            (0, 0),
                            (-1, -1),
                            6
                        ),
                        (
                            "BOTTOMPADDING",
                            (0, 0),
                            (-1, -1),
                            6
                        )
                    ]
                )
            )

            story.append(similar_table)

        story.append(Spacer(1, 20))

        disclaimer = (
            "<b>Research Prototype:</b> "
            "This system is intended for research and "
            "demonstration purposes only. It is not intended "
            "for clinical diagnosis or treatment decisions."
        )

        disclaimer_table = Table(
            [
                [
                    Paragraph(
                        disclaimer,
                        self.center_style
                    )
                ]
            ],
            colWidths=[6.4 * inch]
        )

        disclaimer_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, -1),
                        colors.HexColor("#fff7ed")
                    ),
                    (
                        "BOX",
                        (0, 0),
                        (-1, -1),
                        1,
                        colors.HexColor("#fed7aa")
                    ),
                    (
                        "LEFTPADDING",
                        (0, 0),
                        (-1, -1),
                        10
                    ),
                    (
                        "RIGHTPADDING",
                        (0, 0),
                        (-1, -1),
                        10
                    ),
                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        10
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        10
                    )
                ]
            )
        )

        story.append(disclaimer_table)

        document.build(story)

        pdf_bytes = pdf_buffer.getvalue()
        pdf_buffer.close()

        if output_path:

            output_path = Path(output_path)

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            output_path.write_bytes(pdf_bytes)

        return pdf_bytes