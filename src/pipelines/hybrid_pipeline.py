from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.engine.umap_reducer import umap_reducer
from src.engine.hdbscan import assign_cluster
from src.engine.ArcFace import generate_arcface_embedding
from src.engine.RetinaFace import detect_faces
from src.utils.logger import log
import cv2
import numpy as np
from src.database.face_repository import match_face_embedding_by_cluster

def sign_in(
    *,
    image: Any,
    anomaly_detector: Any,
    threshold: float | None = None,
    match_count: int = 5,
) -> dict[str, Any]:
    faces = detect_faces(image)
    crop = faces[0]["crop"]

    embedding = generate_arcface_embedding(crop)
    print(f"Embeggind producido: {embedding}")
    embedding_reduced = umap_reducer(embedding)
    print(f"Embedding reducido {embedding_reduced}")

    embedding_reduced = np.array(embedding_reduced)
    cluster_id, probability, top_clusters = assign_cluster(embedding_reduced)
    print(f"Asignacion de clusters: cluster_id: {cluster_id}, probability : {probability}")
    print(f"top_clusters: {top_clusters}")

    if probability >= 0.5:
        print(f"Resultado con buena probabilidad ")
        results = match_face_embedding_by_cluster(embedding, cluster_id)
        print(f"Resultados de la consulta en BD: {results}")
    else:
        print(f"Resultado con baja probabilidad ")
        for cluster in top_clusters:
            results = match_face_embedding_by_cluster(embedding, cluster["cluster_id"])
            print(f"Consulta contra el cluster {cluster["cluster_id"]}")
            print(f"Resultados de consulta {results}")
            if results["best_match"] and results["best_match"]["distance"] <= 0.5:
                print(f"Se hayo una relacion con el cluster")
                break
        
        results = match_face_embedding_by_cluster(embedding, -1)
        print(f"Resultado contra los registros sin clusters: {results}")

    return results