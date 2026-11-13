from io import BytesIO
from pathlib import Path
import base64
import math

import cv2
import mediapipe as mp
import numpy as np

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

from measurement.segmentation import (
    SegmentationError,
    segment_body,
)

from measurement.calibration import (
    CalibrationError,
    calculate_pixel_scale,
)

from fastapi.middleware.cors import CORSMiddleware

from measurement.measurements import (
    MeasurementCalculationError,
    calculate_experimental_measurements,
)

from measurement.contours import (
    ContourMeasurementError,
    calculate_contour_measurements,
)


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="ABS CUT AI Measurement Service",
    description="AI-assisted body measurement service for ABS CUT.",
    version="1.9.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ============================================================
# CONFIGURATION
# ============================================================

MAX_IMAGE_SIZE = 10 * 1024 * 1024

MIN_IMAGE_WIDTH = 256
MIN_IMAGE_HEIGHT = 256

MIN_LANDMARK_VISIBILITY = 0.70

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}

MIN_HEIGHT_CM = 40.0
MAX_HEIGHT_CM = 250.0

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "pose_landmarker_lite.task"
)

CORE_BODY_GROUPS = [
    "shoulders",
    "hips",
    "knees",
    "ankles",
]

OPTIONAL_BODY_GROUPS = [
    "elbows",
    "wrists",
]

# ============================================================
# GARMENT CONFIGURATION
# ============================================================

GARMENT_TYPES = {
    "all": "All Garments",
    "agbada": "Agbada",
    "trouser": "Trouser",
    "shirt": "Shirt",
    "senator": "Senator",
    "kaftan": "Kaftan",
    "native_wear": "Native Wear",
    "jacket": "Jacket",
    "waistcoat": "Waistcoat",
    "shorts": "Shorts",
}

# ============================================================
# MEDIAPIPE CONFIGURATION
# ============================================================

BaseOptions = mp.tasks.BaseOptions

PoseLandmarker = mp.tasks.vision.PoseLandmarker

PoseLandmarkerOptions = (
    mp.tasks.vision.PoseLandmarkerOptions
)

VisionRunningMode = mp.tasks.vision.RunningMode


# ============================================================
# HEIGHT VALIDATION AND NORMALIZATION
# ============================================================

def normalize_height(height: float, height_unit: str):

    if not math.isfinite(height):
        raise HTTPException(
            status_code=400,
            detail="Height must be a valid finite number.",
        )

    unit = height_unit.strip().lower()

    if unit not in {"cm", "inches"}:
        raise HTTPException(
            status_code=400,
            detail="Height unit must be either 'cm' or 'inches'.",
        )

    if height <= 0:
        raise HTTPException(
            status_code=400,
            detail="Height must be greater than zero.",
        )

    if unit == "inches":
        height_cm = height * 2.54
    else:
        height_cm = height

    if not MIN_HEIGHT_CM <= height_cm <= MAX_HEIGHT_CM:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Height must be between {MIN_HEIGHT_CM} cm "
                f"and {MAX_HEIGHT_CM} cm."
            ),
        )

    return {
        "entered_value": height,
        "entered_unit": unit,
        "height_cm": round(height_cm, 2),
        "calibration_status": (
            "reference_received_pending_calibration"
        ),
    }

# ============================================================
# GARMENT TYPE VALIDATION
# ============================================================

def normalize_garment_type(garment_type: str):

    if not isinstance(garment_type, str):
        raise HTTPException(
            status_code=400,
            detail="Garment type must be a valid string.",
        )

    normalized = garment_type.strip().lower()

    if normalized not in GARMENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported garment type. "
                f"Supported types: {', '.join(GARMENT_TYPES.keys())}."
            ),
        )

    return normalized

# ============================================================
# IMAGE VALIDATION
# ============================================================

def validate_image_bytes(image_bytes: bytes):

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty.",
        )

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()

        with Image.open(BytesIO(image_bytes)) as image:

            image_format = image.format

            if image_format not in ALLOWED_FORMATS:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Unsupported image format: "
                        f"{image_format or 'unknown'}. "
                        "Only JPEG, PNG and WEBP are accepted."
                    ),
                )

            corrected_image = ImageOps.exif_transpose(image)

            width, height = corrected_image.size

            if (
                width < MIN_IMAGE_WIDTH
                or height < MIN_IMAGE_HEIGHT
            ):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Image resolution is too small. "
                        "Please upload a clearer image."
                    ),
                )

            return {
                "format": image_format,
                "width": width,
                "height": height,
            }

    except HTTPException:
        raise

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    except OSError:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image could not be processed.",
        )


# ============================================================
# POSE DETECTION
# ============================================================

def detect_pose(image_bytes: bytes):

    if not MODEL_PATH.is_file():
        raise HTTPException(
            status_code=500,
            detail=(
                "Pose detection model is missing. "
                "Check models/pose_landmarker_lite.task."
            ),
        )

    try:
        with Image.open(BytesIO(image_bytes)) as image:

            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")

            image_array = np.asarray(
                image,
                dtype=np.uint8,
            )

            image_array = np.ascontiguousarray(image_array)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_array,
        )

        options = PoseLandmarkerOptions(
          base_options=BaseOptions(
    model_asset_buffer=MODEL_PATH.read_bytes()

            ),
            running_mode=VisionRunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        with PoseLandmarker.create_from_options(
            options
        ) as landmarker:

            result = landmarker.detect(mp_image)

        return result

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Pose detection failed: {str(error)}",
        )


# ============================================================
# LANDMARK CHECKS
# ============================================================

def check_landmark(landmarks, landmark_id):

    point = landmarks[landmark_id]

    visibility = float(
        getattr(point, "visibility", 0.0) or 0.0
    )

    inside_image = (
        0 <= point.x <= 1
        and 0 <= point.y <= 1
    )

    visible = (
        visibility >= MIN_LANDMARK_VISIBILITY
    )

    return {
        "visible": visible,
        "inside_image": inside_image,
        "passed": visible and inside_image,
        "visibility": round(visibility, 4),
        "x": round(point.x, 6),
        "y": round(point.y, 6),
    }


def check_body_group(landmarks, left_id, right_id):

    left = check_landmark(
        landmarks,
        left_id,
    )

    right = check_landmark(
        landmarks,
        right_id,
    )

    return {
        "left": left,
        "right": right,
        "passed": (
            left["passed"]
            and right["passed"]
        ),
        "both_sides_passed": (
            left["passed"]
            and right["passed"]
        ),
        "at_least_one_passed": (
            left["passed"]
            or right["passed"]
        ),
    }


# ============================================================
# SIDE-VIEW LANDMARK SELECTION
# ============================================================

def select_best_landmark(group_check):

    candidates = []

    for side in ("left", "right"):

        landmark = group_check[side]

        if landmark["passed"]:

            candidates.append({
                "side": side,
                "visibility": landmark["visibility"],
                "x": landmark["x"],
                "y": landmark["y"],
            })

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda item: item["visibility"],
    )


def add_side_view_selection(checks):

    for group_name in checks:

        group = checks[group_name]

        selected = select_best_landmark(group)

        group["selected_landmark"] = selected

        group["available_sides"] = [
            side
            for side in ("left", "right")
            if group[side]["passed"]
        ]

        group["visible_side_count"] = len(
            group["available_sides"]
        )

        group["both_sides_passed"] = (
            group["left"]["passed"]
            and group["right"]["passed"]
        )

        group["passed"] = selected is not None


# ============================================================
# LANDMARK CONFIDENCE SUMMARY
# ============================================================

def get_landmark_confidence(checks):

    confidence = {}

    for group_name, group in checks.items():

        left_passed = group["left"]["passed"]
        right_passed = group["right"]["passed"]

        if left_passed and right_passed:
            status = "both_visible"

        elif left_passed or right_passed:
            status = "partial"

        else:
            status = "unavailable"

        best_visibility = max(
            group["left"]["visibility"],
            group["right"]["visibility"],
        )

        confidence[group_name] = {
            "status": status,
            "best_visibility": round(
                best_visibility,
                4,
            ),
            "both_sides_visible": (
                left_passed and right_passed
            ),
            "at_least_one_side_visible": (
                left_passed or right_passed
            ),
        }

    return confidence


# ============================================================
# PHOTO QUALITY ASSESSMENT
# ============================================================

def assess_photo_quality(landmarks, view="front"):

    if view not in {"front", "side"}:
        raise ValueError(
            "Photo view must be either 'front' or 'side'."
        )

    checks = {
        "shoulders": check_body_group(
            landmarks, 11, 12
        ),

        "elbows": check_body_group(
            landmarks, 13, 14
        ),

        "wrists": check_body_group(
            landmarks, 15, 16
        ),

        "hips": check_body_group(
            landmarks, 23, 24
        ),

        "knees": check_body_group(
            landmarks, 25, 26
        ),

        "ankles": check_body_group(
            landmarks, 27, 28
        ),
    }

    if view == "front":

        for group_name in CORE_BODY_GROUPS:

            checks[group_name]["passed"] = (
                checks[group_name]["left"]["passed"]
                and checks[group_name]["right"]["passed"]
            )

    else:
        add_side_view_selection(checks)

    passed_groups = sum(
        1
        for group_name in CORE_BODY_GROUPS
        if checks[group_name]["passed"]
    )

    total_groups = len(CORE_BODY_GROUPS)

    quality_score = round(
        (passed_groups / total_groups) * 100
    )

    quality_passed = (
        passed_groups == total_groups
    )

    optional_groups = {}

    for group_name in OPTIONAL_BODY_GROUPS:

        group = checks[group_name]

        optional_groups[group_name] = {
            "both_sides_passed": (
                group["left"]["passed"]
                and group["right"]["passed"]
            ),
            "at_least_one_passed": (
                group["left"]["passed"]
                or group["right"]["passed"]
            ),
            "left": group["left"],
            "right": group["right"],
        }

    failed_groups = [
        group_name
        for group_name in CORE_BODY_GROUPS
        if not checks[group_name]["passed"]
    ]

    recommendations = []

    if failed_groups:

        recommendations.append(
            "Make sure your entire body is visible in the photograph."
        )

        if "shoulders" in failed_groups:
            recommendations.append(
                "Keep your shoulder area clearly visible."
            )

        if "hips" in failed_groups:
            recommendations.append(
                "Make sure your hip area is not obstructed."
            )

        if "knees" in failed_groups:
            recommendations.append(
                "Keep your knee area clearly visible."
            )

        if "ankles" in failed_groups:
            recommendations.append(
                "Keep your ankles and feet inside the photograph."
            )

        recommendations.append(
            "Use good lighting and avoid objects that obstruct your body."
        )

        recommendations.append(
            "Stand upright in the requested photograph position."
        )

    if quality_passed:

        photo_quality = "good"

        message = (
            f"The {view} photograph passed the essential "
            "full-body landmark visibility checks."
        )

    elif passed_groups >= 2:

        photo_quality = "needs_retake"

        message = (
            f"The {view} photograph passed some essential "
            "landmark checks, but other important areas "
            "need improvement."
        )

    else:

        photo_quality = "poor"

        message = (
            f"The {view} photograph is not suitable for "
            "the initial full-body landmark checks."
        )

    landmark_confidence = get_landmark_confidence(
        checks
    )

    return {
        "photo_quality": photo_quality,
        "quality_passed": quality_passed,
        "quality_score": quality_score,
        "quality_score_scope": (
            "Essential body landmark groups only: "
            "shoulders, hips, knees and ankles."
        ),
        "passed_core_groups": passed_groups,
        "total_core_groups": total_groups,
        "failed_core_groups": failed_groups,
        "message": message,
        "checks": checks,
        "landmark_confidence": landmark_confidence,
        "optional_landmarks": optional_groups,
        "recommendations": recommendations,
    }


# ============================================================
# FORMAT LANDMARKS
# ============================================================

def format_landmarks(landmarks):

    return [
        {
            "id": index,
            "x": round(point.x, 6),
            "y": round(point.y, 6),
            "z": round(point.z, 6),
            "visibility": round(
                float(
                    getattr(point, "visibility", 0.0) or 0.0
                ),
                4,
            ),
        }
        for index, point in enumerate(landmarks)
    ]


# ============================================================
# SHARED UPLOAD READER
# ============================================================

async def read_uploaded_image(file: UploadFile):

    try:

        contents = await file.read(
            MAX_IMAGE_SIZE + 1
        )

        if len(contents) > MAX_IMAGE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="Image size must not exceed 10 MB.",
            )

        image_info = validate_image_bytes(contents)

        return contents, image_info

    finally:
        await file.close()


# ============================================================
# IMAGE PREPARATION FOR SEGMENTATION
# ============================================================

def prepare_segmentation_image(contents: bytes):

    try:

        with Image.open(BytesIO(contents)) as image:

            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")

            image_array = np.asarray(
                image,
                dtype=np.uint8,
            )

            return np.ascontiguousarray(image_array)

    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unable to prepare image for segmentation: "
                f"{str(error)}"
            ),
        )


# ============================================================
# SHARED SEGMENTATION PROCESSOR
# ============================================================

def run_body_segmentation(
    contents: bytes,
    include_mask: bool = False,
):

    try:

        image_rgb = prepare_segmentation_image(contents)

        result = segment_body(image_rgb)

        if not include_mask:
            result.pop("mask", None)

        return result

    except SegmentationError as error:

        raise HTTPException(
            status_code=422,
            detail=str(error),
        )

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Body segmentation failed: {str(error)}",
        )


# ============================================================
# IMAGE ENCODING
# ============================================================

def encode_image_data_uri(
    image: Image.Image,
    image_format: str,
):

    buffer = BytesIO()

    save_options = {
        "format": image_format,
        "optimize": True,
    }

    if image_format == "JPEG":
        save_options["quality"] = 90

    image.save(
        buffer,
        **save_options,
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    mime_type = (
        "image/jpeg"
        if image_format == "JPEG"
        else "image/png"
    )

    return {
        "mime_type": mime_type,
        "data_uri": f"data:{mime_type};base64,{encoded}",
    }


# ============================================================
# CREATE SEGMENTATION INSPECTION PREVIEWS
# ============================================================

def create_segmentation_previews(
    contents: bytes,
    mask_png_base64: str,
):

    try:

        with Image.open(BytesIO(contents)) as image:

            original = ImageOps.exif_transpose(image)
            original = original.convert("RGB")

            original_array = np.asarray(
                original,
                dtype=np.uint8,
            )

        mask_bytes = base64.b64decode(
            mask_png_base64,
            validate=True,
        )

        with Image.open(BytesIO(mask_bytes)) as mask_image:

            mask_image = mask_image.convert("L")

            mask_array = np.asarray(
                mask_image,
                dtype=np.uint8,
            )

        if mask_array.shape != original_array.shape[:2]:
            raise ValueError(
                "Segmentation mask dimensions do not match "
                "the original photograph."
            )

        binary_mask = (
            mask_array >= 128
        ).astype(np.uint8)

        alpha_mask = (
            binary_mask * 255
        ).astype(np.uint8)

        original_preview = Image.fromarray(
            original_array,
            mode="RGB",
        )

        rgba_array = np.dstack(
            (
                original_array,
                alpha_mask,
            )
        )

        transparent_preview = Image.fromarray(
            rgba_array,
            mode="RGBA",
        )

        overlay_array = original_array.copy()

        green = np.array(
            [11, 61, 46],
            dtype=np.float32,
        )

        foreground = binary_mask.astype(bool)

        blended = (
            original_array.astype(np.float32) * 0.55
            + green * 0.45
        )

        overlay_array[foreground] = (
            blended[foreground]
        ).clip(0, 255).astype(np.uint8)

        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        overlay_array = cv2.drawContours(
            overlay_array,
            contours,
            -1,
            (201, 162, 39),
            thickness=2,
        )

        overlay_preview = Image.fromarray(
            overlay_array,
            mode="RGB",
        )

        return {
            "original": encode_image_data_uri(
                original_preview,
                "JPEG",
            ),
            "transparent_body_mask": encode_image_data_uri(
                transparent_preview,
                "PNG",
            ),
            "overlay": encode_image_data_uri(
                overlay_preview,
                "JPEG",
            ),
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to create segmentation inspection "
                f"previews: {str(error)}"
            ),
        )


# ============================================================
# SHARED PHOTO ANALYSIS
# ============================================================

def analyze_photo(
    contents: bytes,
    image_info: dict,
    view: str,
):

    result = detect_pose(contents)

    if not result.pose_landmarks:

        return {
            "pose_detected": False,
            "photo_quality": "poor",
            "quality_passed": False,
            "quality_score": 0,
            "quality_score_scope": (
                "Essential body landmark groups only."
            ),
            "passed_core_groups": 0,
            "total_core_groups": len(CORE_BODY_GROUPS),
            "failed_core_groups": CORE_BODY_GROUPS,
            "message": (
                f"No clear human pose was detected in the "
                f"{view} photograph."
            ),
            "landmark_count": 0,
            "landmarks": [],
            "checks": {},
            "landmark_confidence": {},
            "optional_landmarks": {},
            "recommendations": [
                "Make sure a person is clearly visible.",
                "Ensure the entire body is inside the photograph.",
                "Use good lighting.",
                "Retake the photograph in the requested position.",
            ],
            "image": image_info,
            "measurement_status": "not_calculated",
        }

    landmarks = result.pose_landmarks[0]

    quality = assess_photo_quality(
        landmarks,
        view=view,
    )

    return {
        "pose_detected": True,
        "photo_quality": quality["photo_quality"],
        "quality_passed": quality["quality_passed"],
        "quality_score": quality["quality_score"],
        "quality_score_scope": quality["quality_score_scope"],
        "passed_core_groups": quality["passed_core_groups"],
        "total_core_groups": quality["total_core_groups"],
        "failed_core_groups": quality["failed_core_groups"],
        "message": quality["message"],
        "landmark_count": len(landmarks),
        "landmarks": format_landmarks(landmarks),
        "checks": quality["checks"],
        "landmark_confidence": quality["landmark_confidence"],
        "optional_landmarks": quality["optional_landmarks"],
        "recommendations": quality["recommendations"],
        "image": image_info,
        "measurement_status": "not_calculated",
    }


# ============================================================
# MEASUREMENT LANDMARK READINESS
# ============================================================

def build_measurement_landmark_readiness(
    front_analysis: dict,
    side_analysis: dict,
):

    front_checks = front_analysis.get(
        "checks",
        {},
    )

    side_checks = side_analysis.get(
        "checks",
        {},
    )

    def front_pair_available(group_name):

        group = front_checks.get(group_name)

        if not group:
            return False

        return (
            group["left"]["passed"]
            and group["right"]["passed"]
        )

    def side_anchor_available(group_name):

        group = side_checks.get(group_name)

        if not group:
            return False

        return (
            group["left"]["passed"]
            or group["right"]["passed"]
        )

    return {
        "front_view": {
            "shoulders_both_sides": front_pair_available(
                "shoulders"
            ),
            "hips_both_sides": front_pair_available(
                "hips"
            ),
            "knees_both_sides": front_pair_available(
                "knees"
            ),
            "ankles_both_sides": front_pair_available(
                "ankles"
            ),
            "elbows_both_sides": front_pair_available(
                "elbows"
            ),
            "wrists_both_sides": front_pair_available(
                "wrists"
            ),
        },

        "side_view": {
            "shoulder_anchor": side_anchor_available(
                "shoulders"
            ),
            "hip_anchor": side_anchor_available(
                "hips"
            ),
            "knee_anchor": side_anchor_available(
                "knees"
            ),
            "ankle_anchor": side_anchor_available(
                "ankles"
            ),
            "elbow_anchor": side_anchor_available(
                "elbows"
            ),
            "wrist_anchor": side_anchor_available(
                "wrists"
            ),
        },

        "measurement_status": "not_calculated",

        "note": (
            "This report describes landmark availability only. "
            "It does not confirm that any tailoring measurement "
            "has been calculated or validated."
        ),
    }


# ============================================================
# HOME
# ============================================================

@app.get("/")
async def home():

    return {
        "success": True,
        "service": "ABS CUT AI Measurement Service",
        "status": "running",
        "version": "1.9.0",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health():

    model_available = MODEL_PATH.is_file()

    segmentation_available = (
        hasattr(mp, "solutions")
        and hasattr(mp.solutions, "selfie_segmentation")
    )

    return {
        "success": True,
        "status": (
            "healthy"
            if model_available and segmentation_available
            else "degraded"
        ),
        "pose_model_available": model_available,
        "segmentation_model_available": segmentation_available,
    }


# ============================================================
# BASIC IMAGE VALIDATION
# ============================================================

@app.post("/api/measurements/validate-image")
async def validate_image(
    file: UploadFile = File(...),
):

    _, image_info = await read_uploaded_image(file)

    return {
        "success": True,
        "message": "Image passed basic validation.",
        "image": image_info,
        "measurement_status": "not_calculated",
    }


# ============================================================
# BODY SEGMENTATION
# ============================================================

@app.post("/api/measurements/segment-body")
async def segment_body_endpoint(
    file: UploadFile = File(...),
):

    contents, image_info = await read_uploaded_image(file)

    segmentation = run_body_segmentation(contents)

    return {
        "success": True,
        "message": "Body segmentation completed.",
        "image": image_info,
        "segmentation": segmentation,
        "measurement_status": "not_calculated",
    }


# ============================================================
# VISUAL SEGMENTATION INSPECTION
# ============================================================

@app.post("/api/measurements/inspect-segmentation")
async def inspect_segmentation_endpoint(
    file: UploadFile = File(...),
):

    contents, image_info = await read_uploaded_image(file)

    segmentation = run_body_segmentation(contents)

    mask_png_base64 = segmentation.get(
        "mask_png_base64"
    )

    if not mask_png_base64:
        raise HTTPException(
            status_code=500,
            detail=(
                "Segmentation completed without producing "
                "a usable mask image."
            ),
        )

    previews = create_segmentation_previews(
        contents,
        mask_png_base64,
    )

    segmentation.pop(
        "mask_png_base64",
        None,
    )

    return {
        "success": True,
        "message": (
            "Visual segmentation inspection completed."
        ),
        "image": image_info,
        "segmentation": segmentation,
        "previews": previews,
        "inspection_status": (
            "passed_initial_checks"
            if segmentation["quality_passed"]
            else "visual_review_required"
        ),
        "measurement_status": "not_calculated",
    }


# ============================================================
# HEIGHT CALIBRATION
# ============================================================

@app.post("/api/measurements/calibrate-height")
async def calibrate_height_endpoint(

    file: UploadFile = File(...),

    height: float = Form(...),

    height_unit: str = Form(...),

):

    height_reference = normalize_height(
        height,
        height_unit,
    )

    contents, image_info = await read_uploaded_image(file)

    segmentation = run_body_segmentation(contents)

    quality_passed = segmentation["quality_passed"]

    if not quality_passed:

        return {
            "success": True,
            "calibration_status": (
                "not_calibrated_quality_checks_failed"
            ),
            "message": (
                "The silhouette did not pass the initial "
                "segmentation quality checks. Please retake "
                "the photograph before attempting calibration."
            ),
            "height_reference": height_reference,
            "image": image_info,
            "segmentation": segmentation,
            "calibration": None,
            "measurement_status": "not_calculated",
        }

    try:

        silhouette_pixel_height = round(
            segmentation["bounding_box"]["height"]
            * image_info["height"]
        )

        calibration = calculate_pixel_scale(
            height_cm=height_reference["height_cm"],
            silhouette_pixel_height=silhouette_pixel_height,
        )

    except CalibrationError as error:

        raise HTTPException(
            status_code=422,
            detail=str(error),
        )

    return {
        "success": True,
        "calibration_status": "initial_scale_estimated",
        "message": (
            "An initial pixel-to-centimetre scale was estimated "
            "from the supplied height and segmented silhouette. "
            "This calibration has not yet been validated."
        ),
        "height_reference": height_reference,
        "image": image_info,
        "segmentation": {
            "segmentation_model": segmentation["segmentation_model"],
            "experimental": segmentation["experimental"],
            "foreground_ratio": segmentation["foreground_ratio"],
            "bounding_box": segmentation["bounding_box"],
            "quality_passed": segmentation["quality_passed"],
            "quality_flags": segmentation["quality_flags"],
            "mask_png_base64": segmentation["mask_png_base64"],
        },
        "calibration": calibration,
        "measurement_status": "not_calculated",
    }


# ============================================================
# SINGLE PHOTO POSE DETECTION
# PRESERVED FOR BACKWARD COMPATIBILITY
# ============================================================

@app.post("/api/measurements/detect-pose")
async def detect_pose_endpoint(
    file: UploadFile = File(...),
):

    contents, image_info = await read_uploaded_image(file)

    analysis = analyze_photo(
        contents,
        image_info,
        view="front",
    )

    return {
        "success": True,
        **analysis,
    }


# ============================================================
# TWO-PHOTO POSE DETECTION + HEIGHT REFERENCE
# ============================================================

@app.post("/api/measurements/detect-pose-pair")
async def detect_pose_pair(

    front_photo: UploadFile = File(...),

    side_photo: UploadFile = File(...),

    height: float = Form(...),

    height_unit: str = Form(...),

):

    height_reference = normalize_height(
        height,
        height_unit,
    )

    front_contents, front_image_info = (
        await read_uploaded_image(front_photo)
    )

    side_contents, side_image_info = (
        await read_uploaded_image(side_photo)
    )

    front_analysis = analyze_photo(
        front_contents,
        front_image_info,
        view="front",
    )

    side_analysis = analyze_photo(
        side_contents,
        side_image_info,
        view="side",
    )

    both_photos_passed = (
        front_analysis["pose_detected"]
        and side_analysis["pose_detected"]
        and front_analysis["quality_passed"]
        and side_analysis["quality_passed"]
    )

    landmark_readiness = (
        build_measurement_landmark_readiness(
            front_analysis,
            side_analysis,
        )
    )

    if both_photos_passed:

        message = (
            "Both photographs passed the essential "
            "pose and full-body landmark checks. "
            "The height reference was also received. "
            "Measurement calculations have not yet started."
        )

    else:

        failed_views = []

        if not front_analysis["quality_passed"]:
            failed_views.append("front")

        if not side_analysis["quality_passed"]:
            failed_views.append("side")

        message = (
            "The photographs were received, but the "
            f"{' and '.join(failed_views)} photograph"
            f"{'s' if len(failed_views) > 1 else ''} "
            "did not pass all essential landmark checks."
        )

    return {
        "success": True,
        "service_version": "1.9.0",
        "both_photos_passed": both_photos_passed,
        "measurement_status": "not_calculated",
        "message": message,
        "height_reference": height_reference,
        "photos": {
            "front": front_analysis,
            "side": side_analysis,
        },
        "landmark_readiness": landmark_readiness,
        "next_step": (
            "ready_for_measurement_pipeline"
            if both_photos_passed
            else "retake_required"
        ),
    }


# ============================================================
# EXPERIMENTAL BODY MEASUREMENT CALCULATION
# ============================================================

@app.post("/api/measurements/calculate")
async def calculate_measurements_endpoint(

    front_photo: UploadFile = File(...),

    side_photo: UploadFile = File(...),

    height: float = Form(...),

    height_unit: str = Form(...),

    garment_type: str = Form("all"),

):
    garment_type = normalize_garment_type(garment_type)

    # --------------------------------------------------------
    # 1. VALIDATE HEIGHT
    # --------------------------------------------------------

    height_reference = normalize_height(
        height,
        height_unit,
    )

    height_cm = height_reference["height_cm"]

    # --------------------------------------------------------
    # 2. READ BOTH PHOTOGRAPHS
    # --------------------------------------------------------
    # --------------------------------------------------------
    # 2. READ BOTH PHOTOGRAPHS
    # --------------------------------------------------------

    front_contents, front_image_info = (
        await read_uploaded_image(front_photo)
    )

    side_contents, side_image_info = (
        await read_uploaded_image(side_photo)
    )

    # --------------------------------------------------------
    # 3. ANALYSE FRONT PHOTO
    # --------------------------------------------------------

    front_analysis = analyze_photo(
        front_contents,
        front_image_info,
        view="front",
    )

    # --------------------------------------------------------
    # 4. ANALYSE SIDE PHOTO
    # --------------------------------------------------------

    side_analysis = analyze_photo(
        side_contents,
        side_image_info,
        view="side",
    )

    # --------------------------------------------------------
    # 5. VALIDATE POSE QUALITY
    # --------------------------------------------------------

    if not front_analysis["quality_passed"]:

        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    "The front photograph did not pass "
                    "the essential landmark checks."
                ),
                "view": "front",
                "failed_groups": (
                    front_analysis["failed_core_groups"]
                ),
                "recommendations": (
                    front_analysis["recommendations"]
                ),
            },
        )

    if not side_analysis["quality_passed"]:

        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    "The side photograph did not pass "
                    "the essential landmark checks."
                ),
                "view": "side",
                "failed_groups": (
                    side_analysis["failed_core_groups"]
                ),
                "recommendations": (
                    side_analysis["recommendations"]
                ),
            },
        )

    # --------------------------------------------------------
    # 6. RUN FRONT SEGMENTATION
    # Keep the raw mask for internal contour calculations.
    # --------------------------------------------------------

    front_segmentation = run_body_segmentation(
        front_contents,
        include_mask=True,
    )

    # --------------------------------------------------------
    # 7. RUN SIDE SEGMENTATION
    # --------------------------------------------------------

    side_segmentation = run_body_segmentation(
        side_contents,
        include_mask=True,
    )

    # --------------------------------------------------------
    # 8. EXTRACT RAW MASKS
    # --------------------------------------------------------

    front_mask = front_segmentation.pop(
        "mask",
        None,
    )

    side_mask = side_segmentation.pop(
        "mask",
        None,
    )

    if front_mask is None or side_mask is None:

        raise HTTPException(
            status_code=500,
            detail=(
                "Segmentation completed without returning "
                "the internal masks required for contour analysis."
            ),
        )

    # --------------------------------------------------------
    # 9. VALIDATE SEGMENTATION QUALITY
    # --------------------------------------------------------

    if not front_segmentation.get("quality_passed"):

        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    "Front photograph segmentation "
                    "did not pass quality checks."
                ),
                "quality_flags": (
                    front_segmentation.get("quality_flags", [])
                ),
            },
        )

    if not side_segmentation.get("quality_passed"):

        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    "Side photograph segmentation "
                    "did not pass quality checks."
                ),
                "quality_flags": (
                    side_segmentation.get("quality_flags", [])
                ),
            },
        )

    # --------------------------------------------------------
    # 10. EXTRACT SILHOUETTE HEIGHTS
    # --------------------------------------------------------

    try:

        front_bbox = front_segmentation["bounding_box"]
        side_bbox = side_segmentation["bounding_box"]

        front_bbox_height = float(
            front_bbox["height"]
        )

        side_bbox_height = float(
            side_bbox["height"]
        )

        if (
            not math.isfinite(front_bbox_height)
            or not math.isfinite(side_bbox_height)
            or front_bbox_height <= 0
            or side_bbox_height <= 0
        ):
            raise ValueError(
                "Invalid silhouette bounding-box height."
            )

        front_silhouette_height = round(
            front_bbox_height * front_image_info["height"]
        )

        side_silhouette_height = round(
            side_bbox_height * side_image_info["height"]
        )

        if (
            front_silhouette_height <= 0
            or side_silhouette_height <= 0
        ):
            raise ValueError(
                "Calculated silhouette height is invalid."
            )

    except (KeyError, TypeError, ValueError) as error:

        raise HTTPException(
            status_code=422,
            detail=(
                "Unable to extract valid silhouette "
                f"dimensions: {str(error)}"
            ),
        )

    # --------------------------------------------------------
    # 11. CALCULATE EXPERIMENTAL LANDMARK MEASUREMENTS
    # --------------------------------------------------------

    try:

        measurements = calculate_experimental_measurements(

            front_landmarks=front_analysis["landmarks"],

            side_landmarks=side_analysis["landmarks"],

            front_image_width=front_image_info["width"],

            front_image_height=front_image_info["height"],

            side_image_width=side_image_info["width"],

            side_image_height=side_image_info["height"],

            height_cm=height_cm,

            front_silhouette_height=(
                front_silhouette_height
            ),

            side_silhouette_height=(
                side_silhouette_height
            ),
        )

    except MeasurementCalculationError as error:

        raise HTTPException(
            status_code=422,
            detail=str(error),
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Experimental measurement calculation failed: "
                f"{str(error)}"
            ),
        )

    # --------------------------------------------------------
    # 12. CALCULATE EXPERIMENTAL CONTOUR MEASUREMENTS
    # --------------------------------------------------------

    try:

        contour_measurements = calculate_contour_measurements(
            front_mask=front_mask,
            side_mask=side_mask,
            height_cm=height_cm,
        )

    except ContourMeasurementError as error:

        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    "Contour measurement calculation "
                    "could not be completed."
                ),
                "reason": str(error),
            },
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Experimental contour calculation failed: "
                f"{str(error)}"
            ),
        )

    # --------------------------------------------------------
    # 13. RETURN RESULTS
    # --------------------------------------------------------

    return {
        "success": True,

        "service_version": "1.9.0",

        "message": (
            "Experimental 2D body-segment estimates and "
            "silhouette-based circumference approximations "
            "were generated. These values have not been "
            "validated for tailoring accuracy."
        ),

        "measurement_status": (
            "experimental_estimates_generated"
        ),

        "accuracy_status": "not_validated",

        "height_reference": height_reference,
        
        "garment": {
    "type": garment_type,
    "label": GARMENT_TYPES[garment_type],
},

        "photo_validation": {
            "front": {
                "quality_passed": (
                    front_analysis["quality_passed"]
                ),
                "quality_score": (
                    front_analysis["quality_score"]
                ),
            },

            "side": {
                "quality_passed": (
                    side_analysis["quality_passed"]
                ),
                "quality_score": (
                    side_analysis["quality_score"]
                ),
            },
        },

        "segmentation": {
            "front": {
                "quality_passed": (
                    front_segmentation["quality_passed"]
                ),
                "bounding_box": (
                    front_segmentation["bounding_box"]
                ),
            },

            "side": {
                "quality_passed": (
                    side_segmentation["quality_passed"]
                ),
                "bounding_box": (
                    side_segmentation["bounding_box"]
                ),
            },
        },

        "measurements": measurements,

        "contour_measurements": contour_measurements,

        "important_notice": (
            "All values are experimental estimates. "
            "The contour-based circumference values are "
            "derived from approximate front and side silhouette "
            "spans using an elliptical model. They are not verified "
            "body measurements and must not be used as production-ready "
            "tailoring measurements without validation against "
            "manual measurements."
        ),
    }