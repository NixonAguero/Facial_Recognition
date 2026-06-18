from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.pipelines.standard_pipeline import FacePipelineError
from src.engine.umap_reducer import umap_reducer
from src.engine.hdbscan import assign_cluster
from src.engine.ArcFace import generate_arcface_embedding
from src.engine.RetinaFace import detect_faces
from src.utils.logger import log
import cv2
import numpy as np
from src.database.face_repository import (
    match_face_embedding_by_cluster
)
from src.utils.normalizer import (
    l2_normalize_embedding
)
from src.utils.constants import (MIN_HEIGHT, MIN_WIDTH)


class AnomalyDetectedError(FacePipelineError):
    pass


def reduce_embedding(embedding: list[float]) -> list[float]:
    return umap_reducer(embedding)


def analyze_embedding(
    embedding: list[float],
    anomaly_detector: Any,
) -> dict[str, Any]:
    if anomaly_detector is None:
        raise FacePipelineError("No se cargó el autoencoder.")

    try:
        log("Ejecutando UMAP y detector de anomalías...")
        with ThreadPoolExecutor(max_workers=2) as executor:
            umap_task = executor.submit(reduce_embedding, embedding)
            anomaly_task = executor.submit(anomaly_detector.score, embedding)
            reduced_embedding = umap_task.result()
            anomaly_score = float(anomaly_task.result())
    except FileNotFoundError as error:
        raise FacePipelineError("No se encontró el modelo UMAP.") from error

    threshold = float(anomaly_detector.threshold)
    if anomaly_score > threshold:
        raise AnomalyDetectedError(
            f"Rostro rechazado: score={anomaly_score:.6f}, "
            f"umbral={threshold:.6f}"
        )

    return {
        "anomaly_score": anomaly_score,
        "anomaly_threshold": threshold,
        "umap_embedding": reduced_embedding,
        "umap_dimensions": len(reduced_embedding),
    }


def sign_up(
    *,
    images: list[Any],
    filenames: list[str],
    full_name: str,
    external_code: str,
    enrollment_strategy: str,
    anomaly_detector: Any,
) -> dict[str, Any]:
    samples = process_registration_images(images, filenames)
    enrollment_embeddings = select_enrollment_embeddings(
        samples,
        enrollment_strategy,
    )
    analyses = [
        analyze_embedding(embedding, anomaly_detector)
        for embedding in enrollment_embeddings
    ]
    result = save_registration(
        full_name,
        external_code,
        samples,
        enrollment_strategy,
        enrollment_embeddings,
    )
    result["pipeline"] = "hybrid"
    result["quality"] = [sample["quality"] for sample in samples]
    result["hybrid_analysis"] = analyses
    return result


def sign_in(
    *,
    image: Any,
    anomaly_detector: Any,
    threshold: float | None = None,
    match_count: int = 5,
) -> dict[str, Any]:

    faces = detect_faces(image)
    crop = faces[0]["crop"]

    height, weight = crop.shape[:2]
    if height < MIN_HEIGHT and weight < MIN_WIDTH:
        print("Se detecto un recorte con dimensiones mínimas a las permitidas para generar un embedding")
        failed_detection += 1
        return None

    embedding = generate_arcface_embedding(crop)
    print(f"Embeggind producido: {embedding}")
    embedding_normalized = l2_normalize_embedding(embedding)
    print(f"Embedding normalizado {embedding_normalized}")
    embedding_reduced = umap_reducer(embedding_normalized)
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