import os
import sys
from pathlib import Path
import ast

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.database.face_repository import (get_embeddings_registered, save_clusters)
from src.engine.umap_reducer import umap_reducer
from src.engine.hdbscan import train_and_save_hdbscan
import numpy as np
from src.utils.normalizer import (
    l2_normalize_embedding
)

def embedding_reducer(embeddings_to_reduce):

    embeddings_reduced = []

    for embedding in embeddings_to_reduce:

        new_embedding = umap_reducer(embedding)
        embeddings_reduced.append(new_embedding)

    return embeddings_reduced

def parse_embedding(emb):
    if isinstance(emb, str):
        # ast.literal_eval convierte el string "[-0.01, 0.05...]" en una lista real [ -0.01, 0.05 ]
        return ast.literal_eval(emb)
    return emb  

if __name__ == "__main__":

    embeddings_registered = get_embeddings_registered()

    if not embeddings_registered:
        sys.exit(1) 

    embedding_ids = [row["id"] for row in embeddings_registered]
    embeddings_to_arrays = np.array([parse_embedding(row["embedding"]) for row in embeddings_registered])

    embeddings_reduced = embedding_reducer(embeddings_to_arrays)
    embeddings_reduced = np.array(embeddings_reduced)

    cluster_map = train_and_save_hdbscan(embeddings_reduced, embedding_ids) 

    print("Updating cluster_id in database...")

    if not cluster_map:
        print("HDBSCAN training failed.")
        sys.exit(1)

    print(f"cluster map {cluster_map}")

    save_clusters(cluster_map)

    

# import cv2
# from src.database.storage_repository import upload_face_image
# from src.engine.RetinaFace import detect_faces
# from src.engine.ArcFace import generate_arcface_embedding
# from src.database.face_repository import (
#     get_or_create_profile,
#     save_face_image,
#     save_face_embedding,
# )

# def get_best_face_crop(image_path: str) -> np.ndarray | None:

#     print(f"Processing: {image_path}")
#     image = cv2.imread(image_path)

#     if image is None:
#         print(f"  [ERROR] No se pudo leer: {image_path}")
#         return None

#     faces = detect_faces(image)

#     print(f"  [FACES] Detectadas: {len(faces)}")

#     if not faces:
#         print(f"  [INFO] No se detectaron rostros válidos en: {image_path}")
#         return None

#     crop = faces[0]["crop"]
#     print(f"  [CROP] Shape: {crop.shape}")

#     return crop

# def get_image_paths(limite: int = 500) -> list[str]:
#     dataset_dir = Path(PROJECT_ROOT) / "dataset" / "img_align_celeba"

#     if not dataset_dir.exists():
#         sys.exit(1)

#     extensions = ["*.jpg", "*.jpeg", "*.png"]
#     images_paths = []

#     for ext in extensions:
#         for path in dataset_dir.rglob(ext):
#             rel_path = path.relative_to(Path(PROJECT_ROOT))
#             images_paths.append(str(rel_path))
#             if len(images_paths) >= limite:
#                 return images_paths
#     return images_paths

# if __name__ == "__main__":

#     #images_to_register = get_image_paths(limite=500)

#     images_to_register = ["dataset/test/nixon4.png", "dataset/test/cristel4.png", "dataset/test/dereck5.png", "dataset/test/ian2.png", "dataset/test/isma1.png"]

#     print(f"Total a procesar: {len(images_to_register)}")

#     for rel_path in images_to_register:
#         # Usamos el nombre del archivo como nombre y código externo
#         file_name = Path(rel_path).stem
        
#         print(f"\n--- Registrando: {file_name} ---")

#         # 1. Procesar rostro
#         crop = get_best_face_crop(rel_path)
#         if crop is None:
#             continue

#         # 2. Generar embedding
#         embedding = generate_arcface_embedding(crop)
#         if embedding is None:
#             continue

#        embedding_normalized = l2_normalize_embedding(embedding)
        
#         # Convertir a lista para base de datos
#         embedding_list = embedding_normalized.tolist() if hasattr(embedding_normalized, "tolist") else embedding_normalized

#         # 3. Registrar en base de datos
#         profile = get_or_create_profile(full_name=file_name, external_code=file_name)
        
#         # 4. Subir imagen a storage (asumiendo que quieres guardar la original)
#         abs_path = os.path.join(PROJECT_ROOT, rel_path)
#         storage_path = upload_face_image(profile_id=profile["id"], image_path=abs_path)
        
#         # 5. Guardar metadatos y embedding
#         face_image = save_face_image(
#             profile_id=profile["id"], 
#             image_path=storage_path, 
#             image_type="training"
#         )
        
#         save_face_embedding(
#             profile_id=profile["id"], 
#             image_id=face_image["id"], 
#             embedding=embedding_list
#         )
        
#         print(f"Registro exitoso para: {file_name}")