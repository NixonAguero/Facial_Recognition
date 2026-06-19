import os
import sys
import cv2
import numpy as np
from pathlib import Path


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Use the same import style as your project.
# If your project uses "src.engine.RetinaFace", change this line.
from engine.RetinaFace import detect_faces


def bbox_to_xyxy(bbox: dict, image_shape: tuple[int, int, int]) -> tuple[int, int, int, int] | None:
    """
    Converts bbox dictionary into x1, y1, x2, y2 coordinates.

    Supports:
    1. {"x1": ..., "y1": ..., "x2": ..., "y2": ...}
    2. {"x": ..., "y": ..., "width": ..., "height": ...}
    3. {"left": ..., "top": ..., "width": ..., "height": ...}
    """

    h, w = image_shape[:2]

    # Format 1: x1, y1, x2, y2
    if all(key in bbox for key in ["x1", "y1", "x2", "y2"]):
        x1 = int(bbox["x1"])
        y1 = int(bbox["y1"])
        x2 = int(bbox["x2"])
        y2 = int(bbox["y2"])

    # Format 2 or 3: x, y, width, height OR left, top, width, height
    else:
        x = bbox.get("x", bbox.get("left"))
        y = bbox.get("y", bbox.get("top"))

        width = bbox.get("width", bbox.get("w"))
        height = bbox.get("height", bbox.get("h"))

        if x is None or y is None or width is None or height is None:
            print(f"[ERROR] Unsupported bbox format: {bbox}")
            return None

        x1 = int(x)
        y1 = int(y)
        x2 = int(x + width)
        y2 = int(y + height)

    # Keep coordinates inside image boundaries
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        print(f"[WARNING] Invalid bbox after correction: {(x1, y1, x2, y2)}")
        return None

    return x1, y1, x2, y2


def crop_from_bbox(image: np.ndarray, bbox: dict) -> np.ndarray | None:
    coords = bbox_to_xyxy(bbox, image.shape)

    if coords is None:
        return None

    x1, y1, x2, y2 = coords

    crop = image[y1:y2, x1:x2]

    if crop is None or crop.size == 0:
        return None

    return crop


def prepare_image_for_saving(image: np.ndarray) -> np.ndarray | None:
    """
    Converts normalized images into visible uint8 images.
    Useful if we ever need to save face['crop'] from detect_faces().
    """

    if image is None or image.size == 0:
        return None

    if image.max() <= 1.0:
        image = (image * 255).astype("uint8")
    elif image.dtype != np.uint8:
        image = image.astype("uint8")

    return image


def crop_my_frames(person_name: str):
    input_dir = Path(PROJECT_ROOT) / "dataset" / "own_faces" / "frames" / person_name
    output_dir = Path(PROJECT_ROOT) / "dataset" / "own_faces" / "cropped" / person_name
    debug_dir = Path(PROJECT_ROOT) / "dataset" / "own_faces" / "debug" / person_name

    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    image_paths = (
        list(input_dir.glob("*.jpg")) +
        list(input_dir.glob("*.jpeg")) +
        list(input_dir.glob("*.png"))
    )

    if not image_paths:
        print(f"[ERROR] No images found in: {input_dir}")
        return

    saved_count = 0
    skipped_count = 0

    for image_path in image_paths:
        print(f"\n[PROCESSING] {image_path.name}")

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[WARNING] Could not read image: {image_path}")
            skipped_count += 1
            continue

        faces = detect_faces(image)

        print(f"[FACES DETECTED] {len(faces)}")

        if not faces:
            print("[INFO] No valid face found.")
            skipped_count += 1
            continue

        # Since detect_faces already sorts by confidence, faces[0] is the best face.
        best_face = faces[0]

        bbox = best_face.get("bbox")

        if bbox is None:
            print("[WARNING] No bbox found. Trying face['crop'] as fallback.")

            crop = best_face.get("crop")
            if crop is None:
                print("[WARNING] Invalid fallback crop.")
                skipped_count += 1
                continue

            crop = prepare_image_for_saving(crop)

            if crop is None:
                print("[WARNING] Invalid fallback crop.")
                skipped_count += 1
                continue

        else:
            # Preferred option: crop from the original image using the bbox.
            # This avoids saving DeepFace normalized black-looking images.
            crop = crop_from_bbox(image, bbox)

            if crop is None:
                print("[WARNING] Invalid crop from bbox.")
                skipped_count += 1
                continue

        crop = prepare_image_for_saving(crop)

        if crop is None:
            print("[WARNING] Could not prepare crop for saving.")
            skipped_count += 1
            continue

        if crop.max() == 0:
            print("[WARNING] Crop is completely black. Skipping.")
            skipped_count += 1
            continue

        saved_count += 1

        output_name = f"{person_name}_crop_{saved_count:03d}.jpg"
        output_path = output_dir / output_name

        success = cv2.imwrite(str(output_path), crop)

        if not success:
            print(f"[ERROR] Could not save crop: {output_path}")
            skipped_count += 1
            continue

        print(f"[SAVED CROP] {output_path}")

        # Save debug image with bbox rectangle
        if bbox is not None:
            coords = bbox_to_xyxy(bbox, image.shape)

            if coords is not None:
                x1, y1, x2, y2 = coords

                debug_image = image.copy()
                cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                debug_path = debug_dir / f"{person_name}_debug_{saved_count:03d}.jpg"
                cv2.imwrite(str(debug_path), debug_image)

                print(f"[SAVED DEBUG] {debug_path}")

    print("\n[DONE]")
    print(f"[SAVED] {saved_count}")
    print(f"[SKIPPED] {skipped_count}")
    print(f"[OUTPUT] {output_dir}")


if __name__ == "__main__":
    crop_my_frames("ian")