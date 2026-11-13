import math
from typing import Any


class MeasurementCalculationError(Exception):
    """Raised when experimental body measurement calculation fails."""


LANDMARKS = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

MIN_VISIBILITY = 0.50


def _get_landmark(
    landmarks: Any,
    landmark_name: str,
) -> dict | None:
    """
    Retrieve a landmark from either a list of dictionaries
    or a MediaPipe-style landmark collection.
    """

    index = LANDMARKS.get(landmark_name)

    if index is None:
        return None

    try:
        landmark = landmarks[index]
    except (IndexError, KeyError, TypeError):
        return None

    try:
        if isinstance(landmark, dict):
            x = float(landmark["x"])
            y = float(landmark["y"])
            visibility = float(
                landmark.get("visibility", 1.0)
            )
        else:
            x = float(landmark.x)
            y = float(landmark.y)
            visibility = float(
                getattr(landmark, "visibility", 1.0)
            )

        if not all(
            math.isfinite(value)
            for value in (x, y, visibility)
        ):
            return None

        if not 0 <= visibility <= 1:
            return None

        return {
            "name": landmark_name,
            "x": x,
            "y": y,
            "visibility": visibility,
        }

    except (ValueError, TypeError, KeyError, AttributeError):
        return None


def _pixel_distance(
    first: dict,
    second: dict,
    image_width: int,
    image_height: int,
) -> float:
    """Calculate Euclidean distance between normalized landmarks."""

    x1 = first["x"] * image_width
    y1 = first["y"] * image_height

    x2 = second["x"] * image_width
    y2 = second["y"] * image_height

    return math.sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    )


def _unavailable(reason: str) -> dict:
    return {
        "available": False,
        "reason": reason,
        "accuracy_status": "not_validated",
    }


def _segment_from_points(
    first: dict | None,
    second: dict | None,
    image_width: int,
    image_height: int,
    cm_per_pixel: float,
) -> dict:
    """Calculate a projected 2D distance between two landmarks."""

    if first is None or second is None:
        return _unavailable(
            "One or both required landmarks were not detected."
        )

    if (
        first["visibility"] < MIN_VISIBILITY
        or second["visibility"] < MIN_VISIBILITY
    ):
        return _unavailable(
            "Landmark visibility is below the required threshold."
        )

    pixel_distance = _pixel_distance(
        first,
        second,
        image_width,
        image_height,
    )

    estimated_cm = pixel_distance * cm_per_pixel

    return {
        "available": True,
        "pixel_distance": round(pixel_distance, 2),
        "estimated_cm": round(estimated_cm, 2),
        "confidence": round(
            min(
                first["visibility"],
                second["visibility"],
            ),
            3,
        ),
        "landmarks": [
            first["name"],
            second["name"],
        ],
        "accuracy_status": "not_validated",
    }


def _segment(
    landmarks: Any,
    first_name: str,
    second_name: str,
    image_width: int,
    image_height: int,
    cm_per_pixel: float,
) -> dict:
    first = _get_landmark(landmarks, first_name)
    second = _get_landmark(landmarks, second_name)

    return _segment_from_points(
        first,
        second,
        image_width,
        image_height,
        cm_per_pixel,
    )


def _average_span(
    landmarks: Any,
    left_name: str,
    right_name: str,
    image_width: int,
    image_height: int,
    cm_per_pixel: float,
) -> dict:
    """
    Calculate a front-view span between corresponding left and
    right landmarks.
    """

    left = _get_landmark(landmarks, left_name)
    right = _get_landmark(landmarks, right_name)

    return _segment_from_points(
        left,
        right,
        image_width,
        image_height,
        cm_per_pixel,
    )


def _best_visible_side_segment(
    landmarks: Any,
    left_first: str,
    left_second: str,
    right_first: str,
    right_second: str,
    image_width: int,
    image_height: int,
    cm_per_pixel: float,
) -> dict:
    """
    Select the more visible side for a side-view measurement.

    This is useful because MediaPipe may not detect the far-side
    joints clearly in a side-view photograph.
    """

    candidates = []

    left_a = _get_landmark(landmarks, left_first)
    left_b = _get_landmark(landmarks, left_second)

    right_a = _get_landmark(landmarks, right_first)
    right_b = _get_landmark(landmarks, right_second)

    if (
        left_a is not None
        and left_b is not None
        and left_a["visibility"] >= MIN_VISIBILITY
        and left_b["visibility"] >= MIN_VISIBILITY
    ):
        score = min(
            left_a["visibility"],
            left_b["visibility"],
        )

        candidates.append(
            (score, "left", left_a, left_b)
        )

    if (
        right_a is not None
        and right_b is not None
        and right_a["visibility"] >= MIN_VISIBILITY
        and right_b["visibility"] >= MIN_VISIBILITY
    ):
        score = min(
            right_a["visibility"],
            right_b["visibility"],
        )

        candidates.append(
            (score, "right", right_a, right_b)
        )

    if not candidates:
        return _unavailable(
            "Neither side has sufficient landmark visibility."
        )

    candidates.sort(
        key=lambda candidate: candidate[0],
        reverse=True,
    )

    score, selected_side, first, second = candidates[0]

    result = _segment_from_points(
        first,
        second,
        image_width,
        image_height,
        cm_per_pixel,
    )

    result["selected_side"] = selected_side
    result["side_selection_confidence"] = round(score, 3)

    return result


def _validate_calibration(
    height_cm: float,
    front_silhouette_height: int,
    side_silhouette_height: int,
) -> tuple[float, float]:
    if not math.isfinite(height_cm) or height_cm <= 0:
        raise MeasurementCalculationError(
            "Height must be a positive finite number."
        )

    if front_silhouette_height <= 0:
        raise MeasurementCalculationError(
            "Invalid front-view silhouette height."
        )

    if side_silhouette_height <= 0:
        raise MeasurementCalculationError(
            "Invalid side-view silhouette height."
        )

    front_cm_per_pixel = (
        height_cm / front_silhouette_height
    )

    side_cm_per_pixel = (
        height_cm / side_silhouette_height
    )

    return front_cm_per_pixel, side_cm_per_pixel


def _validate_image_dimensions(
    image_width: int,
    image_height: int,
    view_name: str,
) -> None:
    if (
        not isinstance(image_width, int)
        or not isinstance(image_height, int)
        or image_width <= 0
        or image_height <= 0
    ):
        raise MeasurementCalculationError(
            f"Invalid {view_name} image dimensions."
        )


def calculate_experimental_measurements(
    front_landmarks: Any,
    side_landmarks: Any,
    front_image_width: int,
    front_image_height: int,
    side_image_width: int,
    side_image_height: int,
    height_cm: float,
    front_silhouette_height: int,
    side_silhouette_height: int,
) -> dict:
    """
    Generate experimental projected 2D body-segment estimates.

    IMPORTANT:
    These are not validated tailoring measurements.

    The calculation uses image-space landmark distances and
    approximate height-based calibration. It does not calculate
    reliable body circumferences.
    """

    _validate_image_dimensions(
        front_image_width,
        front_image_height,
        "front-view",
    )

    _validate_image_dimensions(
        side_image_width,
        side_image_height,
        "side-view",
    )

    front_scale, side_scale = _validate_calibration(
        height_cm,
        front_silhouette_height,
        side_silhouette_height,
    )

    # ---------------------------------------------------------
    # FRONT VIEW
    # ---------------------------------------------------------

    front_view = {
        "shoulder_span": _average_span(
            front_landmarks,
            "left_shoulder",
            "right_shoulder",
            front_image_width,
            front_image_height,
            front_scale,
        ),

        "upper_arm_left": _segment(
            front_landmarks,
            "left_shoulder",
            "left_elbow",
            front_image_width,
            front_image_height,
            front_scale,
        ),

        "upper_arm_right": _segment(
            front_landmarks,
            "right_shoulder",
            "right_elbow",
            front_image_width,
            front_image_height,
            front_scale,
        ),

        "forearm_left": _segment(
            front_landmarks,
            "left_elbow",
            "left_wrist",
            front_image_width,
            front_image_height,
            front_scale,
        ),

        "forearm_right": _segment(
            front_landmarks,
            "right_elbow",
            "right_wrist",
            front_image_width,
            front_image_height,
            front_scale,
        ),

        "hip_span": _average_span(
            front_landmarks,
            "left_hip",
            "right_hip",
            front_image_width,
            front_image_height,
            front_scale,
        ),

        "hip_to_knee_left": _segment(
            front_landmarks,
            "left_hip",
            "left_knee",
            front_image_width,
            front_image_height,
            front_scale,
        ),

        "hip_to_knee_right": _segment(
            front_landmarks,
            "right_hip",
            "right_knee",
            front_image_width,
            front_image_height,
            front_scale,
        ),

        "knee_to_ankle_left": _segment(
            front_landmarks,
            "left_knee",
            "left_ankle",
            front_image_width,
            front_image_height,
            front_scale,
        ),

        "knee_to_ankle_right": _segment(
            front_landmarks,
            "right_knee",
            "right_ankle",
            front_image_width,
            front_image_height,
            front_scale,
        ),
    }

    # ---------------------------------------------------------
    # SIDE VIEW
    # ---------------------------------------------------------

    side_view = {
        "shoulder_to_hip": _best_visible_side_segment(
            side_landmarks,
            "left_shoulder",
            "left_hip",
            "right_shoulder",
            "right_hip",
            side_image_width,
            side_image_height,
            side_scale,
        ),

        "hip_to_knee": _best_visible_side_segment(
            side_landmarks,
            "left_hip",
            "left_knee",
            "right_hip",
            "right_knee",
            side_image_width,
            side_image_height,
            side_scale,
        ),

        "knee_to_ankle": _best_visible_side_segment(
            side_landmarks,
            "left_knee",
            "left_ankle",
            "right_knee",
            "right_ankle",
            side_image_width,
            side_image_height,
            side_scale,
        ),

        "shoulder_to_wrist": _best_visible_side_segment(
            side_landmarks,
            "left_shoulder",
            "left_wrist",
            "right_shoulder",
            "right_wrist",
            side_image_width,
            side_image_height,
            side_scale,
        ),
    }

    return {
        "measurement_status": "experimental_estimates_generated",

        "method": "2D_pose_landmark_euclidean_distances",

        "accuracy_status": "not_validated",

        "calibration": {
            "height_cm": round(height_cm, 2),

            "front": {
                "silhouette_pixel_height": (
                    front_silhouette_height
                ),
                "cm_per_pixel": round(
                    front_scale,
                    6,
                ),
            },

            "side": {
                "silhouette_pixel_height": (
                    side_silhouette_height
                ),
                "cm_per_pixel": round(
                    side_scale,
                    6,
                ),
            },
        },

        "front_view": front_view,

        "side_view": side_view,

        "circumference_measurements": {
            "status": "not_calculated",
            "reason": (
                "Reliable chest, waist, hip and other body "
                "circumferences cannot be established from "
                "these 2D landmark distances alone."
            ),
        },

        "limitations": [
            (
                "Measurements are projected 2D image distances, "
                "not verified physical body measurements."
            ),
            (
                "Height-based calibration assumes the detected "
                "silhouette height represents the person's "
                "actual height."
            ),
            (
                "Camera perspective, posture, clothing, shoes, "
                "and segmentation boundaries can affect results."
            ),
            (
                "Body circumferences require additional validated "
                "depth, multi-view, or inference methods."
            ),
            (
                "Manual tailoring measurements should be used "
                "for verification before garment production."
            ),
        ],
    }



# import math
# from typing import Any


# class MeasurementCalculationError(Exception):
#     """Raised when experimental measurement calculations cannot be completed."""


# # MediaPipe Pose landmark indices
# LANDMARKS = {
#     "left_shoulder": 11,
#     "right_shoulder": 12,
#     "left_elbow": 13,
#     "right_elbow": 14,
#     "left_wrist": 15,
#     "right_wrist": 16,
#     "left_hip": 23,
#     "right_hip": 24,
#     "left_knee": 25,
#     "right_knee": 26,
#     "left_ankle": 27,
#     "right_ankle": 28,
# }


# def _get_landmark(landmarks: Any, index: int) -> dict:
#     """Accept either a MediaPipe landmark list or a list of dictionaries."""
#     try:
#         item = landmarks[index]
#     except (IndexError, KeyError, TypeError):
#         raise MeasurementCalculationError(
#             f"Landmark index {index} is unavailable."
#         )

#     if isinstance(item, dict):
#         x, y = item.get("x"), item.get("y")
#         visibility = item.get("visibility", item.get("presence", 1.0))
#     else:
#         x, y = getattr(item, "x", None), getattr(item, "y", None)
#         visibility = getattr(item, "visibility", 1.0)

#     if x is None or y is None:
#         raise MeasurementCalculationError(
#             f"Landmark index {index} has no coordinates."
#         )

#     x, y = float(x), float(y)
#     visibility = float(visibility or 0.0)

#     if not math.isfinite(x) or not math.isfinite(y):
#         raise MeasurementCalculationError(
#             f"Landmark index {index} has invalid coordinates."
#         )

#     return {"x": x, "y": y, "visibility": visibility}


# def _point(landmarks: Any, name: str) -> dict:
#     return _get_landmark(landmarks, LANDMARKS[name])


# def _pixel_distance(a: dict, b: dict, width: int, height: int) -> float:
#     dx = (a["x"] - b["x"]) * width
#     dy = (a["y"] - b["y"]) * height
#     return math.sqrt(dx * dx + dy * dy)


# def _segment(
#     landmarks: Any,
#     start_name: str,
#     end_name: str,
#     image_width: int,
#     image_height: int,
#     cm_per_pixel: float,
#     minimum_visibility: float = 0.5,
# ) -> dict:
#     start = _point(landmarks, start_name)
#     end = _point(landmarks, end_name)

#     if min(start["visibility"], end["visibility"]) < minimum_visibility:
#         return {
#             "status": "unavailable",
#             "reason": "One or both landmarks have insufficient visibility.",
#         }

#     distance_px = _pixel_distance(
#         start, end, image_width, image_height
#     )

#     return {
#         "status": "experimental_estimate",
#         "pixel_distance": round(distance_px, 2),
#         "estimated_cm": round(distance_px * cm_per_pixel, 2),
#         "landmark_visibility": {
#             "start": round(start["visibility"], 4),
#             "end": round(end["visibility"], 4),
#         },
#         "accuracy_status": "not_validated",
#     }


# def _average_point(landmarks: Any, left_name: str, right_name: str) -> dict:
#     left = _point(landmarks, left_name)
#     right = _point(landmarks, right_name)

#     if min(left["visibility"], right["visibility"]) < 0.5:
#         raise MeasurementCalculationError(
#             f"Both {left_name} and {right_name} must be visible."
#         )

#     return {
#         "x": (left["x"] + right["x"]) / 2,
#         "y": (left["y"] + right["y"]) / 2,
#         "visibility": min(left["visibility"], right["visibility"]),
#     }


# def _average_span(
#     landmarks: Any,
#     left_name: str,
#     right_name: str,
#     image_width: int,
#     image_height: int,
#     cm_per_pixel: float,
# ) -> dict:
#     left = _point(landmarks, left_name)
#     right = _point(landmarks, right_name)

#     if min(left["visibility"], right["visibility"]) < 0.5:
#         return {
#             "status": "unavailable",
#             "reason": "Both landmarks must be visible.",
#         }

#     distance_px = _pixel_distance(left, right, image_width, image_height)

#     return {
#         "status": "experimental_estimate",
#         "pixel_distance": round(distance_px, 2),
#         "estimated_cm": round(distance_px * cm_per_pixel, 2),
#         "accuracy_status": "not_validated",
#     }


# def calculate_experimental_measurements(
#     front_landmarks: Any,
#     front_image_width: int,
#     front_image_height: int,
#     side_landmarks: Any,
#     side_image_width: int,
#     side_image_height: int,
#     height_cm: float,
#     front_silhouette_pixel_height: int,
#     side_silhouette_pixel_height: int,
# ) -> dict:
#     """
#     Produce preliminary 2D landmark segment estimates.

#     Important:
#     - The scale assumes the segmentation silhouette height equals the user's
#       entered physical height.
#     - This assumption is approximate and camera-dependent.
#     - These values are NOT validated tailoring measurements.
#     - No circumference is inferred from a 2D image.
#     """
#     if height_cm <= 0 or not math.isfinite(height_cm):
#         raise MeasurementCalculationError("Height must be positive and finite.")

#     dimensions = (
#         front_image_width,
#         front_image_height,
#         side_image_width,
#         side_image_height,
#         front_silhouette_pixel_height,
#         side_silhouette_pixel_height,
#     )
#     if any(value <= 0 for value in dimensions):
#         raise MeasurementCalculationError(
#             "Image dimensions and silhouette heights must be positive."
#         )

#     front_cm_per_pixel = height_cm / front_silhouette_pixel_height
#     side_cm_per_pixel = height_cm / side_silhouette_pixel_height

#     front = {
#         "shoulder_landmark_span": _average_span(
#             front_landmarks, "left_shoulder", "right_shoulder",
#             front_image_width, front_image_height, front_cm_per_pixel,
#         ),
#         "upper_arm_left": _segment(
#             front_landmarks, "left_shoulder", "left_elbow",
#             front_image_width, front_image_height, front_cm_per_pixel,
#         ),
#         "upper_arm_right": _segment(
#             front_landmarks, "right_shoulder", "right_elbow",
#             front_image_width, front_image_height, front_cm_per_pixel,
#         ),
#         "forearm_left": _segment(
#             front_landmarks, "left_elbow", "left_wrist",
#             front_image_width, front_image_height, front_cm_per_pixel,
#         ),
#         "forearm_right": _segment(
#             front_landmarks, "right_elbow", "right_wrist",
#             front_image_width, front_image_height, front_cm_per_pixel,
#         ),
#         "hip_landmark_span": _average_span(
#             front_landmarks, "left_hip", "right_hip",
#             front_image_width, front_image_height, front_cm_per_pixel,
#         ),
#         "hip_to_knee_left": _segment(
#             front_landmarks, "left_hip", "left_knee",
#             front_image_width, front_image_height, front_cm_per_pixel,
#         ),
#         "hip_to_knee_right": _segment(
#             front_landmarks, "right_hip", "right_knee",
#             front_image_width, front_image_height, front_cm_per_pixel,
#         ),
#         "knee_to_ankle_left": _segment(
#             front_landmarks, "left_knee", "left_ankle",
#             front_image_width, front_image_height, front_cm_per_pixel,
#         ),
#         "knee_to_ankle_right": _segment(
#             front_landmarks, "right_knee", "right_ankle",
#             front_image_width, front_image_height, front_cm_per_pixel,
#         ),
#     }

#     # Side-view anchors are used for a projected shoulder-to-hip torso segment.
#     side = {
#         "shoulder_to_hip": _segment(
#             side_landmarks, "left_shoulder", "left_hip",
#             side_image_width, side_image_height, side_cm_per_pixel,
#         ),
#         "hip_to_knee": _segment(
#             side_landmarks, "left_hip", "left_knee",
#             side_image_width, side_image_height, side_cm_per_pixel,
#         ),
#         "knee_to_ankle": _segment(
#             side_landmarks, "left_knee", "left_ankle",
#             side_image_width, side_image_height, side_cm_per_pixel,
#         ),
#     }

#     return {
#         "measurement_status": "experimental_estimates_generated",
#         "calibration": {
#             "height_cm": round(height_cm, 2),
#             "front_silhouette_pixel_height": front_silhouette_pixel_height,
#             "side_silhouette_pixel_height": side_silhouette_pixel_height,
#             "front_cm_per_pixel": round(front_cm_per_pixel, 6),
#             "side_cm_per_pixel": round(side_cm_per_pixel, 6),
#             "method": "entered_height_divided_by_segmentation_silhouette_height",
#             "accuracy_status": "not_validated",
#         },
#         "front_view": front,
#         "side_view": side,
#         "not_calculated": [
#             "chest_circumference",
#             "waist_circumference",
#             "hip_circumference",
#             "neck_circumference",
#         ],
#         "limitations": [
#             "All segment values are experimental 2D estimates.",
#             "Camera perspective and posture can affect scale and distances.",
#             "Landmark joint centres are not necessarily garment measurement points.",
#             "Circumferences are not calculated from these 2D images.",
#             "Independent manual-measurement validation is required.",
#         ],
#     }
