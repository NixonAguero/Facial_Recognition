import ast
import os
import sys
import cv2
import numpy as np
from pathlib import Path
import umap

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.engine.ArcFace import generate_arcface_embedding
from src.engine.umap_reducer import train_and_save_umap
from src.engine.RetinaFace import detect_faces

from src.database.face_repository import (get_embeddings_registered, save_clusters)

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

    extensions = [".jpg", ".jpeg", "*.png"]
    images_paths = []

    for ext in extensions:
        for path in dataset_dir.rglob(ext):
            rel_path = path.relative_to(Path(PROJECT_ROOT))
            images_paths.append(str(rel_path))

    return images_paths

def parse_embedding(emb):
    if isinstance(emb, str):
        return ast.literal_eval(emb)
    return emb

if __name__ == "__main__":
    collected_embeddings = get_embeddings_registered()
    print(f"Embeddings obtenidos de la base de datos {len(collected_embeddings)}")
    embeddings_numeric = []
    for item in collected_embeddings:
        parsed_val = parse_embedding(item["embedding"])
        embeddings_numeric.append(parsed_val)

    embeddings_matrix = np.array(embeddings_numeric, dtype=np.float32)

    train_and_save_umap(embeddings_matrix)  