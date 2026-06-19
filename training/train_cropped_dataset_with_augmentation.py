import os
import sys
import cv2
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np


# If this script is inside /scripts
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


from src.engine.RetinaFace import detect_faces


PERSON_NAME = "ian"

CELEBA_LIMIT = 5000
OWN_IMAGES_LIMIT = None

MAX_WORKERS = 4

CLEAN_OUTPUT = True

SOURCE_CELEBA_DIR = Path(PROJECT_ROOT) / "dataset" / "img_align_celeba"
SOURCE_OWN_DIR = Path(PROJECT_ROOT) / "dataset" / "own_faces" / "cropped" / PERSON_NAME

CROPPED_OUTPUT_DIR = Path(PROJECT_ROOT) / "dataset" / "training_cropped_parallel"


def get_image_paths_from_folder(folder: Path, limit: int | None = None) -> list[Path]:
    extensions = ["*.jpg", "*.jpeg", "*.png"]
    image_paths = []

    if not folder.exists():
        print(f"[WARNING] Folder does not exist: {folder}")
        return image_paths

    for ext in extensions:
        for path in folder.rglob(ext):
            image_paths.append(path)

            if limit is not None and len(image_paths) >= limit:
                return image_paths

    return image_paths


def get_all_source_images() -> list[Path]:
    image_paths = []

    image_paths.extend(get_image_paths_from_folder(SOURCE_CELEBA_DIR, CELEBA_LIMIT))
    image_paths.extend(get_image_paths_from_folder(SOURCE_OWN_DIR, OWN_IMAGES_LIMIT))

    return image_paths


def get_source_label(image_path: Path) -> str:
    parts = image_path.parts

    if PERSON_NAME in parts:
        return PERSON_NAME

    if "img_align_celeba" in parts:
        return "img_align_celeba"

    return image_path.parent.name


def prepare_image_for_saving(image: np.ndarray | None) -> np.ndarray | None:
    if image is None or image.size == 0:
        return None

    if image.max() <= 1.0:
        image = (image * 255).astype("uint8")
    elif image.dtype != np.uint8:
        image = image.astype("uint8")

    return image


def bbox_to_xyxy(bbox: dict[str, Any], image_shape: tuple[int, int, int]) -> tuple[int, int, int, int] | None:
    h, w = image_shape[:2]

    if all(key in bbox for key in ["x1", "y1", "x2", "y2"]):
        x1 = int(bbox["x1"])
        y1 = int(bbox["y1"])
        x2 = int(bbox["x2"])
        y2 = int(bbox["y2"])
    else:
        x = bbox.get("x", bbox.get("left"))
        y = bbox.get("y", bbox.get("top"))
        width = bbox.get("width", bbox.get("w"))
        height = bbox.get("height", bbox.get("h"))

        if x is None or y is None or width is None or height is None:
            return None

        x1 = int(x)
        y1 = int(y)
        x2 = int(x + width)
        y2 = int(y + height)

    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def crop_from_bbox(image: np.ndarray, bbox: dict[str, Any]) -> np.ndarray | None:
    coords = bbox_to_xyxy(bbox, image.shape)

    if coords is None:
        return None

    x1, y1, x2, y2 = coords

    crop = image[y1:y2, x1:x2]

    if crop is None or crop.size == 0:
        return None

    return crop


def crop_single_image(image_path_str: str) -> dict:
    """
    This function runs in a separate process.
    It must receive simple serializable values.
    """

    image_path = Path(image_path_str)

    try:
        image = cv2.imread(str(image_path))

        if image is None:
            return {
                "status": "skipped",
                "reason": "could_not_read",
                "input": str(image_path),
            }

        faces = detect_faces(image)

        if not faces:
            return {
                "status": "skipped",
                "reason": "no_face_detected",
                "input": str(image_path),
            }

        best_face = faces[0]
        bbox = best_face.get("bbox")

        if bbox is not None:
            crop = crop_from_bbox(image, bbox)
        else:
            crop = best_face.get("crop")

        crop = prepare_image_for_saving(crop)

        if crop is None or crop.size == 0:
            return {
                "status": "skipped",
                "reason": "invalid_crop",
                "input": str(image_path),
            }

        if crop.max() == 0:
            return {
                "status": "skipped",
                "reason": "black_crop",
                "input": str(image_path),
            }

        source_label = get_source_label(image_path)
        output_subdir = CROPPED_OUTPUT_DIR / source_label
        output_subdir.mkdir(parents=True, exist_ok=True)

        output_name = f"{image_path.stem}_crop.jpg"
        output_path = output_subdir / output_name

        saved = cv2.imwrite(str(output_path), crop)

        if not saved:
            return {
                "status": "skipped",
                "reason": "could_not_save",
                "input": str(image_path),
            }

        return {
            "status": "success",
            "input": str(image_path),
            "output": str(output_path),
            "crop_shape": crop.shape,
        }

    except Exception as e:
        return {
            "status": "error",
            "reason": str(e),
            "input": str(image_path),
        }


def parallel_crop_dataset():
    if CLEAN_OUTPUT and CROPPED_OUTPUT_DIR.exists():
        print(f"[INFO] Cleaning output folder: {CROPPED_OUTPUT_DIR}")
        shutil.rmtree(CROPPED_OUTPUT_DIR)

    CROPPED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = get_all_source_images()

    print(f"[PROJECT_ROOT] {PROJECT_ROOT}")
    print(f"[CELEBA] {SOURCE_CELEBA_DIR} exists={SOURCE_CELEBA_DIR.exists()}")
    print(f"[OWN] {SOURCE_OWN_DIR} exists={SOURCE_OWN_DIR.exists()}")
    print(f"[TOTAL INPUT IMAGES] {len(image_paths)}")
    print(f"[MAX_WORKERS] {MAX_WORKERS}")

    if not image_paths:
        print("[ERROR] No images found.")
        return

    success_count = 0
    skipped_count = 0
    error_count = 0

    image_path_strings = [str(path) for path in image_paths]

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(crop_single_image, image_path)
            for image_path in image_path_strings
        ]

        total = len(futures)

        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()

            status = result.get("status")

            if status == "success":
                success_count += 1
            elif status == "skipped":
                skipped_count += 1
            else:
                error_count += 1

            if index % 25 == 0 or index == total:
                print("\n---------- CROPPING PROGRESS ----------")
                print(f"Processed: {index}/{total}")
                print(f"Success: {success_count}")
                print(f"Skipped: {skipped_count}")
                print(f"Errors: {error_count}")
                print("--------------------------------------")

            if status != "success":
                print(f"[{(status or 'unknown').upper()}] {result.get('reason')} | {result.get('input')}")

    print("\n[CROPPING DONE]")
    print(f"[SUCCESS] {success_count}")
    print(f"[SKIPPED] {skipped_count}")
    print(f"[ERRORS] {error_count}")
    print(f"[OUTPUT] {CROPPED_OUTPUT_DIR}")


if __name__ == "__main__":
    ##############parallel_crop_dataset()