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
from src.utils.constants import (MIN_HEIGHT, MIN_WIDTH)
from src.utils.capture_frame import capture_frame
from src.engine.anomaly_detector import AnomalyDetector
from src.utils.normalizer import (
    calculate_embedding_centroid,
    l2_normalize_embedding,
)

def sign_in(
    *,
    image: Any,
    anomaly_detector: Any,
    threshold: float | None = None,
    match_count: int = 5,
) -> dict[str, Any]:

    if image is None:
        print("No hay imagen que detectar")
        return None

    try:
        faces = detect_faces(image)
    except Exception as error:
        error_detail = str(error).splitlines()[0]
        raise FacePipelineError(
            f"RetinaFace no pudo procesar la imagen: {error_detail}"
        ) from error

    if not faces:
        raise FacePipelineError("No se detecto un rostro valido.")

    if len(faces) > 1:
        raise FacePipelineError("Se detecto mas de un rostro.")

    crop = faces[0]["crop"]

    height, weight = crop.shape[:2]
    if height < MIN_HEIGHT and weight < MIN_WIDTH:
        print("Se detecto un recorte con dimensiones mínimas a las permitidas para generar un embedding")
        return None

    embedding = generate_arcface_embedding(crop)

    try:
        normalized_embedding = l2_normalize_embedding(embedding)
    except ValueError as error:
        raise FacePipelineError(str(error)) from error

    if anomaly_detector.is_anomaly(normalized_embedding):
        print("Se detecto una anomalia en el embedding producido, la imagen ingresada no es un rostro valido")
        return None

    print(f"Embeggind producido: {normalized_embedding}")
    embedding_reduced = umap_reducer(normalized_embedding)
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