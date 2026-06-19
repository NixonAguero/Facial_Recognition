import os
import sys
import shutil
from pathlib import Path

import cv2
import numpy as np
import albumentations as A


# ============================================================
# PROJECT ROOT
# ============================================================

# Use this if this script is inside /scripts
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Use this instead if the script is directly in project root:
# PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.engine.ArcFace import generate_arcface_embedding
from src.database.storage_repository import upload_face_image
from src.database.face_repository import (
    get_or_create_profile,
    save_face_image,
    save_face_embedding,
)
from src.utils.normalizer import l2_normalize_embedding


# ============================================================
# CONFIG
# ============================================================

PERSON_NAME = "ian"

# Folder where your cropped images are stored.
# Use the one that exists in your project.
CROPPED_INPUT_DIR = Path(PROJECT_ROOT) / "dataset" / "training_cropped_parallel"

# If you used the non-parallel script, use this instead:
# CROPPED_INPUT_DIR = Path(PROJECT_ROOT) / "dataset" / "training_cropped_again"

AUGMENTED_OUTPUT_DIR = Path(PROJECT_ROOT) / "dataset" / "training_cropped_augmented"

AUGMENTATIONS_PER_IMAGE = 1

CLEAN_AUGMENTED_OUTPUT = True

# Use a small limit first to test.
# Then change to None when everything works.
PROCESS_LIMIT = None
# PROCESS_LIMIT = None


# ============================================================
# AUGMENTATION
# ============================================================

def build_light_augmentation_pipeline():
    return A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.6
        ),

        A.Rotate(
            limit=8,
            p=0.4
        ),

        A.GaussianBlur(
            blur_limit=(3, 3),
            p=0.15
        ),

        A.GaussNoise(
            p=0.15
        ),

        A.ImageCompression(
            quality_range=(80, 100),
            p=0.25
        )
    ])


# ============================================================
# UTILS
# ============================================================

def get_all_image_paths(folder: Path, limit: int | None = None) -> list[Path]:
    extensions = ["*.jpg", "*.jpeg", "*.png"]
    image_paths = []

    if not folder.exists():
        print(f"[ERROR] Folder does not exist: {folder}")
        return image_paths

    for ext in extensions:
        for path in folder.rglob(ext):
            image_paths.append(path)

            if limit is not None and len(image_paths) >= limit:
                return image_paths

    return image_paths


def get_source_label(image_path: Path) -> str:
    """
    Keeps the same group/folder structure.

    Examples:
    dataset/training_cropped_parallel/img_align_celeba/000001_crop.jpg
        -> img_align_celeba

    dataset/training_cropped_parallel/ian/ian_crop_001.jpg
        -> ian
    """

    if PERSON_NAME in image_path.parts:
        return PERSON_NAME

    if "img_align_celeba" in image_path.parts:
        return "img_align_celeba"

    return image_path.parent.name


def get_profile_data_from_path(rel_path: str) -> tuple[str, str]:
    """
    Your own images should all go to the same profile.

    Example:
    dataset/training_cropped_augmented/ian/ian_crop_001_aug_1.jpg
        -> full_name = ian
        -> external_code = ian

    Other dataset images keep filename behavior.
    """

    path = Path(rel_path)

    if PERSON_NAME in path.parts:
        return PERSON_NAME, PERSON_NAME

    file_name = path.stem
    return file_name, file_name


# ============================================================
# STEP 1: AUGMENT CROPPED IMAGES
# ============================================================

def augment_cropped_images():
    print("\n===================================================")
    print("[STEP 1] AUGMENTING CROPPED IMAGES")
    print("===================================================")

    print(f"[INPUT] {CROPPED_INPUT_DIR}")
    print(f"[OUTPUT] {AUGMENTED_OUTPUT_DIR}")
    print(f"[AUGMENTATIONS_PER_IMAGE] {AUGMENTATIONS_PER_IMAGE}")
    print(f"[PROCESS_LIMIT] {PROCESS_LIMIT}")

    if CLEAN_AUGMENTED_OUTPUT and AUGMENTED_OUTPUT_DIR.exists():
        print(f"[INFO] Cleaning old output folder: {AUGMENTED_OUTPUT_DIR}")
        shutil.rmtree(AUGMENTED_OUTPUT_DIR)

    AUGMENTED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = get_all_image_paths(CROPPED_INPUT_DIR, limit=PROCESS_LIMIT)

    if not image_paths:
        print("[ERROR] No cropped images found.")
        return []

    print(f"[INFO] Total cropped images found: {len(image_paths)}")

    transform = build_light_augmentation_pipeline()

    final_paths = []
    originals_saved = 0
    augmentations_saved = 0
    skipped_count = 0

    for index, image_path in enumerate(image_paths, start=1):
        print(f"\n[AUGMENT {index}/{len(image_paths)}] {image_path.name}")

        image = cv2.imread(str(image_path))

        if image is None:
            print("  [SKIP] Could not read image.")
            skipped_count += 1
            continue

        source_label = get_source_label(image_path)
        output_subdir = AUGMENTED_OUTPUT_DIR / source_label
        output_subdir.mkdir(parents=True, exist_ok=True)

        # Save original cropped image
        original_output_path = output_subdir / f"{image_path.stem}_original.jpg"

        if cv2.imwrite(str(original_output_path), image):
            originals_saved += 1
            final_paths.append(original_output_path)
            print(f"  [SAVED ORIGINAL] {original_output_path.name}")
        else:
            print("  [SKIP] Could not save original.")
            skipped_count += 1
            continue

        # Save augmented versions
        for aug_index in range(1, AUGMENTATIONS_PER_IMAGE + 1):
            augmented = transform(image=image)
            augmented_image = augmented["image"]

            augmented_output_path = output_subdir / f"{image_path.stem}_aug_{aug_index}.jpg"

            if cv2.imwrite(str(augmented_output_path), augmented_image):
                augmentations_saved += 1
                final_paths.append(augmented_output_path)
                print(f"  [SAVED AUG] {augmented_output_path.name}")
            else:
                print(f"  [WARNING] Could not save augmentation {aug_index}")
                skipped_count += 1

        if index % 25 == 0:
            print("\n  -------- AUGMENTATION PROGRESS --------")
            print(f"  Processed: {index}/{len(image_paths)}")
            print(f"  Originals saved: {originals_saved}")
            print(f"  Augmentations saved: {augmentations_saved}")
            print(f"  Skipped: {skipped_count}")
            print("  ---------------------------------------")

    print("\n[AUGMENTATION DONE]")
    print(f"[ORIGINALS SAVED] {originals_saved}")
    print(f"[AUGMENTATIONS SAVED] {augmentations_saved}")
    print(f"[SKIPPED] {skipped_count}")
    print(f"[TOTAL FINAL IMAGES] {len(final_paths)}")

    return final_paths


# ============================================================
# STEP 2: REGISTER IMAGES + EMBEDDINGS
# ============================================================

def register_augmented_images():
    print("\n===================================================")
    print("[STEP 2] REGISTERING AUGMENTED IMAGES + EMBEDDINGS")
    print("===================================================")

    image_paths = get_all_image_paths(AUGMENTED_OUTPUT_DIR, limit=None)

    if not image_paths:
        print("[ERROR] No augmented images found.")
        return

    print(f"[INFO] Total images to register: {len(image_paths)}")

    success_count = 0
    skipped_count = 0

    for index, image_path in enumerate(image_paths, start=1):
        rel_path = str(image_path.relative_to(Path(PROJECT_ROOT)))
        full_name, external_code = get_profile_data_from_path(rel_path)

        print(f"\n[REGISTER {index}/{len(image_paths)}] {rel_path}")
        print(f"  [PROFILE] full_name={full_name} | external_code={external_code}")

        image = cv2.imread(str(image_path))

        if image is None:
            print("  [SKIP] Could not read image.")
            skipped_count += 1
            continue

        print(f"  [IMAGE] Shape: {image.shape}")

        # Images are already cropped, so do NOT run detect_faces here.
        embedding = generate_arcface_embedding(image)

        if embedding is None:
            print("  [SKIP] Could not generate embedding.")
            skipped_count += 1
            continue

        embedding_normalized = l2_normalize_embedding(embedding)

        embedding_list = list(embedding_normalized)

        try:
            profile = get_or_create_profile(
                full_name=full_name,
                external_code=external_code
            )
        except Exception as e:
            print(f"  [SKIP] Could not get/create profile: {e}")
            skipped_count += 1
            continue

        try:
            storage_path = upload_face_image(
                profile_id=profile["id"],
                image_path=str(image_path)
            )
        except Exception as e:
            print(f"  [SKIP] Could not upload image to storage: {e}")
            skipped_count += 1
            continue

        try:
            face_image_record = save_face_image(
                profile_id=profile["id"],
                image_path=storage_path,
                image_type="training"
            )
        except Exception as e:
            print(f"  [SKIP] Could not save image metadata: {e}")
            skipped_count += 1
            continue

        try:
            save_face_embedding(
                profile_id=profile["id"],
                image_id=face_image_record["id"],
                embedding=embedding_list
            )
        except Exception as e:
            print(f"  [SKIP] Could not save embedding: {e}")
            skipped_count += 1
            continue

        success_count += 1
        print("  [SUCCESS] Registered image + embedding.")

        if index % 25 == 0:
            print("\n  -------- REGISTRATION PROGRESS --------")
            print(f"  Processed: {index}/{len(image_paths)}")
            print(f"  Success: {success_count}")
            print(f"  Skipped: {skipped_count}")
            print("  ---------------------------------------")

    print("\n[REGISTRATION DONE]")
    print(f"[SUCCESS] {success_count}")
    print(f"[SKIPPED] {skipped_count}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n===================================================")
    print("PIPELINE: AUGMENT + REGISTER CROPPED IMAGES")
    print("===================================================")

    print(f"[PROJECT_ROOT] {PROJECT_ROOT}")
    print(f"[CROPPED_INPUT_DIR] {CROPPED_INPUT_DIR} | exists={CROPPED_INPUT_DIR.exists()}")
    print(f"[AUGMENTED_OUTPUT_DIR] {AUGMENTED_OUTPUT_DIR}")
    print(f"[AUGMENTATIONS_PER_IMAGE] {AUGMENTATIONS_PER_IMAGE}")
    print(f"[PROCESS_LIMIT] {PROCESS_LIMIT}")

    augment_cropped_images()
    register_augmented_images()

    print("\n===================================================")
    print("PIPELINE FINISHED")
    print("===================================================")