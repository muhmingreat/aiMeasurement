
import base64
import io

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image


class SegmentationError(Exception):
    """Raised when body segmentation cannot be completed."""


def decode_image(image_bytes: bytes) -> np.ndarray:
    """
    Decode uploaded image bytes into an RGB NumPy array.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return np.asarray(image)

    except Exception as exc:
        raise SegmentationError(
            "Unable to decode uploaded image."
        ) from exc


def encode_mask_png(mask: np.ndarray) -> str:
    """
    Encode a binary segmentation mask as base64 PNG.
    """
    try:
        mask_image = Image.fromarray(
            (mask * 255).astype(np.uint8)
        )

        buffer = io.BytesIO()
        mask_image.save(buffer, format="PNG")

        return base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

    except Exception as exc:
        raise SegmentationError(
            "Unable to encode segmentation mask."
        ) from exc


def segment_body(image_rgb: np.ndarray) -> dict:
    """
    Generate an experimental person segmentation mask.

    Input:
        RGB image as a NumPy array.

    Output:
        Segmentation mask, bounding box, foreground ratio,
        and basic segmentation quality indicators.

    Important:
        This is an experimental segmentation stage.
        It does not calculate tailoring measurements.
    """

    if image_rgb is None or image_rgb.size == 0:
        raise SegmentationError(
            "The supplied image is empty."
        )

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise SegmentationError(
            "Expected an RGB image."
        )

    height, width = image_rgb.shape[:2]

    if width < 100 or height < 100:
        raise SegmentationError(
            "Image resolution is too small."
        )

    # Generate person segmentation mask.
    try:
        with mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=1
        ) as segmenter:

            result = segmenter.process(image_rgb)

            if result.segmentation_mask is None:
                raise SegmentationError(
                    "The segmentation model did not produce a mask."
                )

            probability_mask = result.segmentation_mask

    except SegmentationError:
        raise

    except Exception as exc:
        raise SegmentationError(
            f"Segmentation failed: {str(exc)}"
        ) from exc

    # Convert probability mask into binary mask.
    binary_mask = (
        probability_mask >= 0.5
    ).astype(np.uint8)

    # Smooth the mask and remove small gaps.
    kernel = np.ones((5, 5), np.uint8)

    binary_mask = cv2.morphologyEx(
        binary_mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    # Identify connected foreground components.
    try:
        num_labels, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                binary_mask,
                connectivity=8,
            )
        )

    except Exception as exc:
        raise SegmentationError(
            "Unable to analyze segmentation components."
        ) from exc

    if num_labels <= 1:
        raise SegmentationError(
            "No foreground person was detected."
        )

    # Select the largest foreground component.
    largest_label = 1 + np.argmax(
        stats[1:, cv2.CC_STAT_AREA]
    )

    body_mask = (
        labels == largest_label
    ).astype(np.uint8)

    # Extract silhouette bounding box.
    x = int(
        stats[largest_label, cv2.CC_STAT_LEFT]
    )

    y = int(
        stats[largest_label, cv2.CC_STAT_TOP]
    )

    box_width = int(
        stats[largest_label, cv2.CC_STAT_WIDTH]
    )

    box_height = int(
        stats[largest_label, cv2.CC_STAT_HEIGHT]
    )

    body_pixels = int(
        stats[largest_label, cv2.CC_STAT_AREA]
    )

    foreground_ratio = (
        body_pixels / float(width * height)
    )

    # Normalize bounding box coordinates.
    bounding_box = {
        "x": round(x / width, 6),
        "y": round(y / height, 6),
        "width": round(box_width / width, 6),
        "height": round(box_height / height, 6),
    }

    # Basic segmentation quality checks.
    quality_flags = []

    if foreground_ratio < 0.05:
        quality_flags.append(
            "foreground_area_too_small"
        )

    if foreground_ratio > 0.90:
        quality_flags.append(
            "foreground_area_unusually_large"
        )

    if box_height / height < 0.40:
        quality_flags.append(
            "person_silhouette_too_short"
        )

    if y <= 1:
        quality_flags.append(
            "silhouette_touches_top_edge"
        )

    if y + box_height >= height - 1:
        quality_flags.append(
            "silhouette_touches_bottom_edge"
        )

    # Encode mask for API preview.
    mask_base64 = encode_mask_png(body_mask)

    return {
        "success": True,
        "segmentation_model": "mediapipe_selfie_segmentation",
        "experimental": True,
        "image_width": width,
        "image_height": height,
        "foreground_pixels": body_pixels,
        "foreground_ratio": round(
            foreground_ratio,
            6,
        ),
        "bounding_box": bounding_box,
        "quality_flags": quality_flags,
        "quality_passed": len(quality_flags) == 0,
        "mask": body_mask,
        "mask_png_base64": mask_base64,
    }
