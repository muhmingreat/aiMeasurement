import math

import numpy as np


class ContourMeasurementError(Exception):
    """Raised when silhouette contour measurements cannot be calculated."""


# Approximate positions measured down from the top of the detected silhouette.
# These are provisional positions, not validated anatomical landmarks.
BODY_LEVELS = {
    "chest": 0.32,
    "waist": 0.46,
    "hip": 0.57,
}


def _prepare_binary_mask(mask):
    if mask is None:
        raise ContourMeasurementError(
            "A segmentation mask is required."
        )

    array = np.asarray(mask)

    if array.ndim == 3:
        if array.shape[2] == 1:
            array = array[:, :, 0]
        else:
            array = array[:, :, 0]

    if array.ndim != 2 or array.size == 0:
        raise ContourMeasurementError(
            "Invalid segmentation mask dimensions."
        )

    if array.dtype == np.bool_:
        binary = array
    else:
        try:
            array = array.astype(np.float32)
        except (TypeError, ValueError):
            raise ContourMeasurementError(
                "Segmentation mask contains unsupported values."
            )

        if not np.isfinite(array).all():
            raise ContourMeasurementError(
                "Segmentation mask contains invalid values."
            )

        maximum = float(array.max())

        if maximum <= 1.0:
            binary = array >= 0.5
        else:
            binary = array >= 128

    if int(binary.sum()) < 100:
        raise ContourMeasurementError(
            "The detected body silhouette is too small."
        )

    return binary.astype(bool)


def _get_silhouette_bounds(mask):
    ys, xs = np.where(mask)

    if len(xs) == 0 or len(ys) == 0:
        raise ContourMeasurementError(
            "No foreground silhouette was detected."
        )

    return {
        "x_min": int(xs.min()),
        "x_max": int(xs.max()),
        "y_min": int(ys.min()),
        "y_max": int(ys.max()),
        "width": int(xs.max() - xs.min() + 1),
        "height": int(ys.max() - ys.min() + 1),
    }


def _foreground_runs(row):
    positions = np.flatnonzero(row)

    if len(positions) == 0:
        return []

    splits = np.where(np.diff(positions) > 1)[0] + 1
    groups = np.split(positions, splits)

    return [
        (int(group[0]), int(group[-1]))
        for group in groups
        if len(group) > 0
    ]


def _measure_horizontal_span(mask, bounds, level_ratio):
    height = bounds["height"]

    center_y = round(
        bounds["y_min"] + level_ratio * (height - 1)
    )

    center_x = (
        bounds["x_min"] + bounds["x_max"]
    ) / 2

    radius = max(2, round(height * 0.01))

    y_start = max(
        bounds["y_min"],
        center_y - radius,
    )

    y_end = min(
        bounds["y_max"],
        center_y + radius,
    )

    widths = []

    for y in range(y_start, y_end + 1):
        runs = _foreground_runs(mask[y])

        if not runs:
            continue

        # Select the foreground run closest to the silhouette centre.
        # This is a heuristic and may still include clothing or limbs.
        selected = min(
            runs,
            key=lambda run: abs(
                ((run[0] + run[1]) / 2) - center_x
            ),
        )

        width = selected[1] - selected[0] + 1

        if width > 0:
            widths.append(width)

    if not widths:
        return None

    return round(float(np.median(widths)), 2)


def _ellipse_circumference(width_cm, depth_cm):
    if (
        not math.isfinite(width_cm)
        or not math.isfinite(depth_cm)
        or width_cm <= 0
        or depth_cm <= 0
    ):
        return None

    a = width_cm / 2
    b = depth_cm / 2

    denominator = a + b

    if denominator <= 0:
        return None

    h = ((a - b) ** 2) / (denominator ** 2)

    circumference = (
        math.pi
        * denominator
        * (
            1
            + (3 * h)
            / (10 + math.sqrt(4 - 3 * h))
        )
    )

    if not math.isfinite(circumference):
        return None

    return round(circumference, 2)


def _measure_view(mask, height_cm, view_name):
    binary = _prepare_binary_mask(mask)
    bounds = _get_silhouette_bounds(binary)

    if (
        not isinstance(height_cm, (int, float))
        or not math.isfinite(height_cm)
        or height_cm <= 0
    ):
        raise ContourMeasurementError(
            "Invalid reference height."
        )

    cm_per_pixel = height_cm / bounds["height"]

    measurements = {}

    for name, level in BODY_LEVELS.items():
        pixel_width = _measure_horizontal_span(
            binary,
            bounds,
            level,
        )

        if pixel_width is None:
            measurements[name] = {
                "available": False,
                "pixel_width": None,
                "estimated_cm": None,
                "reason": (
                    "No reliable silhouette span was found "
                    "at this approximate body level."
                ),
                "accuracy_status": "not_validated",
            }
            continue

        measurements[name] = {
            "available": True,
            "level_ratio": level,
            "pixel_width": pixel_width,
            "estimated_cm": round(
                pixel_width * cm_per_pixel,
                2,
            ),
            "accuracy_status": "not_validated",
        }

    return {
        "view": view_name,
        "silhouette_width_pixels": bounds["width"],
        "silhouette_height_pixels": bounds["height"],
        "cm_per_pixel": round(cm_per_pixel, 6),
        "measurements": measurements,
    }


def calculate_contour_measurements(
    front_mask,
    side_mask,
    height_cm,
):
    """
    Generate experimental torso circumference estimates.

    Front-view silhouette spans are treated as approximate widths.
    Side-view silhouette spans are treated as approximate depths.

    The two dimensions are combined using an elliptical circumference
    approximation.

    This method is not validated for tailoring accuracy.
    """

    front = _measure_view(
        front_mask,
        height_cm,
        "front",
    )

    side = _measure_view(
        side_mask,
        height_cm,
        "side",
    )

    circumference_results = {}

    for body_part in BODY_LEVELS:
        front_result = front["measurements"][body_part]
        side_result = side["measurements"][body_part]

        if (
            not front_result["available"]
            or not side_result["available"]
        ):
            circumference_results[body_part] = {
                "available": False,
                "front_width_cm": (
                    front_result.get("estimated_cm")
                ),
                "side_depth_cm": (
                    side_result.get("estimated_cm")
                ),
                "estimated_cm": None,
                "reason": (
                    "Both front width and side depth "
                    "are required."
                ),
                "accuracy_status": "not_validated",
            }
            continue

        width_cm = front_result["estimated_cm"]
        depth_cm = side_result["estimated_cm"]

        circumference = _ellipse_circumference(
            width_cm,
            depth_cm,
        )

        circumference_results[body_part] = {
            "available": circumference is not None,
            "front_width_cm": width_cm,
            "side_depth_cm": depth_cm,
            "estimated_cm": circumference,
            "method": "ramanujan_ellipse_approximation",
            "accuracy_status": "not_validated",
        }

    return {
        "status": "experimental_contour_estimates_generated",
        "method": (
            "front_side_silhouette_ellipse_approximation"
        ),
        "accuracy_status": "not_validated",
        "front_view": front,
        "side_view": side,
        "estimated_circumferences": circumference_results,
        "limitations": [
            (
                "Silhouette widths may include clothing, arms, "
                "or other foreground regions."
            ),
            (
                "The selected vertical body levels are approximate "
                "and may not match actual anatomical positions."
            ),
            (
                "The ellipse model assumes an approximately "
                "elliptical body cross-section."
            ),
            (
                "Height calibration is affected by camera perspective, "
                "distance, and posture."
            ),
            (
                "Front and side photographs may have different "
                "camera distances and perspectives."
            ),
            (
                "These estimates have not been validated against "
                "manual tailoring measurements."
            ),
        ],
    }