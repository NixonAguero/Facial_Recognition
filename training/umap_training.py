import os
import sys
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.engine.ArcFace import generate_arcface_embedding
from src.engine.umap_reducer import train_and_save_umap
from src.engine.RetinaFace import detect_faces
from src.utils.constants import (MIN_HEIGHT, MIN_WIDTH)

def get_best_face_crop(image_path: str) -> np.ndarray | None:

    print(f"Processing: {image_path}")
    image = cv2.imread(image_path)

    if image is None:
        print(f"  [ERROR] No se pudo leer: {image_path}")
        return None

    faces = detect_faces(image)

    print(f"  [FACES] Detectadas: {len(faces)}")

    if not faces:
        print(f"  [INFO] No se detectaron rostros válidos en: {image_path}")
        return None

    crop = faces[0]["crop"]
    print(f"  [CROP] Shape: {crop.shape}")

    return crop


def get_image_paths() -> list[str]:
    dataset_dir = Path(PROJECT_ROOT) / "dataset" / "img_align_celeba"

    if not dataset_dir.exists():
        print(f"Error: Dataset not found at {dataset_dir}")
        sys.exit(1)

    extensions = ["*.jpg", "*.jpeg", "*.png"]
    images_paths = []

    for ext in extensions:
        for path in dataset_dir.rglob(ext):
            rel_path = path.relative_to(Path(PROJECT_ROOT))
            images_paths.append(str(rel_path))

    return images_paths

if __name__ == "__main__":

    all_image_paths = get_image_paths()  
    print(f"Total images found: {len(all_image_paths)}")

    collected_embeddings = []
    failed_detection = 0
    failed_embedding = 0

    for i, image_path in enumerate(all_image_paths):
        if i % 500 == 0:
            print(
                f"Progress: {i}/{len(all_image_paths)}"
                f" | Embeddings: {len(collected_embeddings)}"
                f" | Failed detection: {failed_detection}"
                f" | Failed embedding: {failed_embedding}"
            )

        crop = get_best_face_crop(image_path)

        if crop is None:
            failed_detection += 1
            continue

        height, weight = crop.shape[:2]
        if height < MIN_HEIGHT and weight < MIN_WIDTH:
            print("Se detecto un recorte con dimensiones mínimas a las permitidas para generar un embedding")
            failed_detection += 1
            continue

        embedding_vector = generate_arcface_embedding(crop)

        if embedding_vector is None:
            failed_embedding += 1
            continue

        collected_embeddings.append(embedding_vector)

    print(f"\n{'='*50}")
    print(f"Total images processed : {len(all_image_paths)}")
    print(f"Embeddings collected   : {len(collected_embeddings)}")
    print(f"Failed detection       : {failed_detection}")
    print(f"Failed embedding       : {failed_embedding}")
    print(f"Success rate           : {len(collected_embeddings)/len(all_image_paths)*100:.1f}%")
    print(f"{'='*50}")

    train_and_save_umap(collected_embeddings)