
import io
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    Table, TableStyle, PageBreak
)


def create_flood_report(
    sar_path,
    probability_path,
    depth_path,
    risk_path,
    rainfall_path,
    rainfall_mm=None
):
    pdf_buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=22,
        leading=26,
        alignment=TA_CENTER,
        spaceAfter=16
    )

    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        spaceBefore=12,
        spaceAfter=8
    )

    normal_style = ParagraphStyle(
        "ReportNormal",
        parent=styles["BodyText"],
        fontSize=9,
        leading=13
    )

    story = []

    # ---------------------------------------------------------
    # READ RASTERS
    # ---------------------------------------------------------

    with rasterio.open(sar_path) as src:
        sar = src.read()
        sar_profile = src.profile

    with rasterio.open(probability_path) as src:
        probability = src.read(1)

    with rasterio.open(depth_path) as src:
        depth = src.read(1)

    with rasterio.open(risk_path) as src:
        risk = src.read(1)

    valid = np.isfinite(sar).all(axis=0)

    probability = np.nan_to_num(
        probability, nan=0.0, posinf=0.0, neginf=0.0
    )

    depth = np.nan_to_num(
        depth, nan=0.0, posinf=0.0, neginf=0.0
    )

    # ---------------------------------------------------------
    # STATISTICS
    # ---------------------------------------------------------

    flood_mask = (probability >= 0.5) & valid
    flood_pixels = int(flood_mask.sum())
    valid_pixels = int(valid.sum())

    flood_percentage = (
        flood_pixels / valid_pixels * 100
        if valid_pixels > 0 else 0
    )

    flood_depth_values = depth[flood_mask]

    if flood_depth_values.size > 0:
        avg_depth = float(flood_depth_values.mean())
        max_depth = float(flood_depth_values.max())
        min_depth = float(flood_depth_values.min())
    else:
        avg_depth = max_depth = min_depth = 0.0

    # ---------------------------------------------------------
    # TITLE
    # ---------------------------------------------------------

    story.append(
        Paragraph(
            "FLOOD INTELLIGENCE",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Geo-Spatial AI & Physics-Informed Flood Analytics",
            ParagraphStyle(
                "Subtitle",
                parent=normal_style,
                alignment=TA_CENTER,
                fontSize=11
            )
        )
    )

    story.append(Spacer(1, 18))

    story.append(
        Paragraph(
            "Satellite-Based Flood Mapping and Continuous Depth Estimation",
            heading_style
        )
    )

    story.append(
        Paragraph(
            "This report presents the flood analysis generated from "
            "Sentinel-1 SAR imagery using a trained U-Net segmentation "
            "model followed by a physics-informed neural network (PINN) "
            "for estimated flood depth and project-defined risk mapping.",
            normal_style
        )
    )

    story.append(Spacer(1, 12))

    # ---------------------------------------------------------
    # EVENT SUMMARY
    # ---------------------------------------------------------

    story.append(
        Paragraph("FLOOD EVENT SUMMARY", heading_style)
    )

    rainfall_text = (
        f"{rainfall_mm:.1f} mm"
        if rainfall_mm is not None else "Available in application"
    )

    summary_data = [
        ["Parameter", "Value"],
        ["Input Data", "Sentinel-1 SAR GeoTIFF"],
        ["SAR Bands", "VV + VH"],
        ["Valid Pixels", f"{valid_pixels:,}"],
        ["Estimated Flood Pixels", f"{flood_pixels:,}"],
        ["Estimated Flood Area", f"{flood_percentage:.2f}%"],
        ["24-hour Rainfall", rainfall_text],
        ["Average Estimated Depth", f"{avg_depth:.3f} m"],
        ["Maximum Estimated Depth", f"{max_depth:.3f} m"],
        ["Minimum Estimated Depth", f"{min_depth:.3f} m"],
    ]

    table = Table(
        summary_data,
        colWidths=[2.5 * inch, 3.5 * inch]
    )

    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123B5D")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#F2F6F9")]),
        ])
    )

    story.append(table)
    story.append(Spacer(1, 16))

    # ---------------------------------------------------------
    # IMAGE GENERATOR
    # ---------------------------------------------------------

    def raster_image(array, title, cmap="viridis"):
        buffer = io.BytesIO()

        fig, ax = plt.subplots(figsize=(7, 4.8))

        ax.imshow(array, cmap=cmap)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

        plt.tight_layout()
        fig.savefig(
            buffer,
            format="png",
            dpi=160,
            bbox_inches="tight"
        )

        plt.close(fig)
        buffer.seek(0)

        return Image(
            buffer,
            width=6.6 * inch,
            height=4.4 * inch
        )

    # ---------------------------------------------------------
    # SAR IMAGE
    # ---------------------------------------------------------

    story.append(
        Paragraph("SATELLITE SAR OBSERVATION", heading_style)
    )

    sar_display = sar[0].copy()

    finite = np.isfinite(sar_display)

    if finite.any():
        low = np.percentile(sar_display[finite], 2)
        high = np.percentile(sar_display[finite], 98)

        sar_display = np.clip(
            sar_display,
            low,
            high
        )

    story.append(
        raster_image(
            sar_display,
            "Sentinel-1 SAR VV",
            "gray"
        )
    )

    story.append(PageBreak())

    # ---------------------------------------------------------
    # FLOOD PROBABILITY
    # ---------------------------------------------------------

    story.append(
        Paragraph("U-NET FLOOD PROBABILITY", heading_style)
    )

    probability_display = probability.copy()
    probability_display[~valid] = np.nan

    story.append(
        raster_image(
            probability_display,
            "Flood Probability",
            "Blues"
        )
    )

    story.append(PageBreak())

    # ---------------------------------------------------------
    # DEPTH
    # ---------------------------------------------------------

    story.append(
        Paragraph("PINN FLOOD DEPTH ESTIMATION", heading_style)
    )

    depth_display = depth.copy()
    depth_display[~flood_mask] = np.nan

    story.append(
        raster_image(
            depth_display,
            "Estimated Flood Depth (m)",
            "Blues"
        )
    )

    story.append(
        Paragraph(
            "Depth values are model-estimated outputs from the "
            "physics-informed neural network. They are not presented "
            "as measured ground-truth water-depth observations.",
            normal_style
        )
    )

    story.append(PageBreak())

    # ---------------------------------------------------------
    # RISK
    # ---------------------------------------------------------

    story.append(
        Paragraph("FLOOD RISK MAP", heading_style)
    )

    risk_display = risk.copy().astype(float)
    risk_display[~flood_mask] = np.nan

    story.append(
        raster_image(
            risk_display,
            "Project-Defined Flood Risk",
            "RdYlGn_r"
        )
    )

    story.append(
        Paragraph(
            "Risk classes are project-defined analytical categories "
            "based on estimated flood depth and are not official "
            "emergency-response thresholds.",
            normal_style
        )
    )

    story.append(PageBreak())

    # ---------------------------------------------------------
    # METHODOLOGY
    # ---------------------------------------------------------

    story.append(
        Paragraph("AI PROCESSING PIPELINE", heading_style)
    )

    pipeline = [
        ["Stage", "Method"],
        ["1", "Sentinel-1 VV/VH SAR preprocessing"],
        ["2", "U-Net flood segmentation"],
        ["3", "Flood probability generation"],
        ["4", "DEM and rainfall integration"],
        ["5", "Physics-informed depth estimation"],
        ["6", "Depth-based project risk classification"],
    ]

    pipeline_table = Table(
        pipeline,
        colWidths=[0.7 * inch, 5.3 * inch]
    )

    pipeline_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123B5D")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )

    story.append(pipeline_table)
    story.append(Spacer(1, 18))

    story.append(
        Paragraph(
            "<b>Important:</b> U-Net flood segmentation performance "
            "was evaluated using the project's labeled test data. "
            "The continuous flood-depth output is a model estimate "
            "for the study scene and should not be interpreted as "
            "field-measured water depth.",
            normal_style
        )
    )

    doc.build(story)

    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()
