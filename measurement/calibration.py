
import math


class CalibrationError(Exception):
    """Raised when height calibration cannot be completed."""


def height_to_cm(height: float, unit: str) -> float:
    """
    Convert a supplied height to centimetres.
    """
    if not math.isfinite(height) or height <= 0:
        raise CalibrationError("Height must be a positive finite number.")

    unit = unit.strip().lower()

    if unit == "cm":
        height_cm = height
    elif unit in ("inch", "inches", "in"):
        height_cm = height * 2.54
    else:
        raise CalibrationError(
            "Height unit must be cm or inches."
        )

    if not 40 <= height_cm <= 250:
        raise CalibrationError(
            "Height must be between 40 and 250 cm."
        )

    return round(height_cm, 2)


def calculate_pixel_scale(
    height_cm: float,
    silhouette_pixel_height: int,
) -> dict:
    """
    Calculate an approximate pixel-to-centimetre scale.

    This assumes the detected silhouette height represents
    the person's actual physical height.
    """
    if not math.isfinite(height_cm) or height_cm <= 0:
        raise CalibrationError("Invalid height in centimetres.")

    if silhouette_pixel_height <= 0:
        raise CalibrationError("Invalid silhouette pixel height.")

    pixels_per_cm = silhouette_pixel_height / height_cm

    cm_per_pixel = height_cm / silhouette_pixel_height

    return {
        "calibration_status": "initial_scale_estimated",
        "height_cm": round(height_cm, 2),
        "silhouette_pixel_height": silhouette_pixel_height,
        "pixels_per_cm": round(pixels_per_cm, 6),
        "cm_per_pixel": round(cm_per_pixel, 6),
        "accuracy_status": "not_validated",
    }


def pixels_to_cm(
    pixel_distance: float,
    cm_per_pixel: float,
) -> float:
    """
    Convert a pixel distance into an approximate centimetre distance.
    """
    if not math.isfinite(pixel_distance) or pixel_distance < 0:
        raise CalibrationError("Pixel distance must be non-negative.")

    if not math.isfinite(cm_per_pixel) or cm_per_pixel <= 0:
        raise CalibrationError("Invalid calibration scale.")

    return round(pixel_distance * cm_per_pixel, 2)
