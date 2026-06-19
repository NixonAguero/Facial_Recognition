from typing import Any

import cv2
import numpy as np
from deepface import DeepFace

from utils.constants import (
    MIN_CONFIDENCE,
    MIN_SHARPNESS,
)


def detect_faces(image: np.ndarray) -> list[dict[str, Any]]:
    """Detect, align and normalize valid faces with RetinaFace."""

    try:
        detected_faces = DeepFace.extract_faces(
            img_path=image,
            detector_backend="retinaface",
            enforce_detection=True,
            align=True,
            color_face="bgr",
            normalize_face=True,
        )
    except Exception as e:
        print(f"Error occurred while detecting faces: {e}")
        return []

    valid_faces = []

    for detected_face in detected_faces:
        facial_area = detected_face["facial_area"]
        confidence = float(detected_face["confidence"])
        bbox = create_bbox(facial_area)
        bbox_area = bbox["width"] * bbox["height"]
        original_crop = crop_image(image, bbox)

        sharpness = calculate_sharpness(original_crop)
        brightness = calculate_brightness(original_crop)
        contrast = calculate_contrast(original_crop)
        landmarks = get_landmarks(facial_area)

        if confidence < MIN_CONFIDENCE:
            print(f"Flag 1: Confidence {confidence} below threshold {MIN_CONFIDENCE}")
            continue
        if sharpness < MIN_SHARPNESS:
            print(f"Flag 2: Sharpness {sharpness} below threshold {MIN_SHARPNESS}")
            continue
        if brightness < 20 or brightness > 220:
            print(f"Flag 3: Brightness {brightness} outside range [20, 220]")
            continue
        if contrast < 20:
            print(f"Flag 4: Contrast {contrast} below threshold 20")
            continue
        if len(landmarks) != 5:
            print(f"Flag 5: The image has {len(landmarks)} landmarks, expected 5, landmarks detected: {facial_area}")
            continue

        valid_faces.append(
            {
                "bbox": bbox,
                "confidence": confidence,
                "sharpness": sharpness,
                "bbox_area_px": bbox_area,
                "landmarks": landmarks,
                "crop": detected_face["face"],
            }
        )

    valid_faces.sort(
        key=lambda face: face["confidence"],
        reverse=True,
    )
    return valid_faces


def create_bbox(facial_area: dict[str, Any]) -> dict[str, float]:
    x1 = float(facial_area["x"])
    y1 = float(facial_area["y"])
    width = float(facial_area["w"])
    height = float(facial_area["h"])

    return {
        "x1": x1,
        "y1": y1,
        "x2": x1 + width,
        "y2": y1 + height,
        "width": width,
        "height": height,
    }


def get_landmarks(facial_area: dict[str, Any]) -> list[dict[str, Any]]:

    print(f"Facial area landmarks: {facial_area}")

    landmark_names = (
        "right_eye",
        "left_eye",
        "nose",
        "mouth_right",
        "mouth_left",
    )

    return [
        {
            "name": name,
            "x": float(facial_area[name][0]),
            "y": float(facial_area[name][1]),
        }
        for name in landmark_names
        if facial_area.get(name) is not None
    ]


def crop_image(
    image: np.ndarray,
    bbox: dict[str, float],
) -> np.ndarray:
    height, width = image.shape[:2]
    x1 = max(0, int(bbox["x1"]))
    y1 = max(0, int(bbox["y1"]))
    x2 = min(width, int(bbox["x2"]))
    y2 = min(height, int(bbox["y2"]))
    return image[y1:y2, x1:x2]


def calculate_sharpness(crop: np.ndarray) -> float:
    if crop.size == 0:
        return 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def calculate_brightness(crop: np.ndarray) -> float:
    if crop.size == 0:
        return 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.mean(gray)[0])


def calculate_contrast(crop: np.ndarray) -> float:
    if crop.size == 0:
        return 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(gray.std())
