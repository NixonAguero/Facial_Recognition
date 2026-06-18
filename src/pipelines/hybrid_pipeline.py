from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from src.database.face_repository import match_face_embedding_by_cluster
from src.engine.ArcFace import generate_arcface_embedding
from src.engine.RetinaFace import detect_faces
from src.engine.hdbscan import assign_cluster
from src.engine.umap_reducer import umap_reducer
from src.pipelines.standard_pipeline import FacePipelineError, save_registration
from src.utils.logger import log
from src.utils.normalizer import (
    calculate_embedding_centroid,
    l2_normalize_embedding,
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
        raise FacePipelineError("No se cargo el autoencoder.")

    try:
        log("Ejecutando UMAP y detector de anomalias...")
        with ThreadPoolExecutor(max_workers=2) as executor:
            umap_task = executor.submit(reduce_embedding, embedding)
            anomaly_task = executor.submit(anomaly_detector.score, embedding)
            reduced_embedding = umap_task.result()
            anomaly_score = float(anomaly_task.result())
    except FileNotFoundError as error:
        raise FacePipelineError("No se encontro el modelo UMAP.") from error

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


def process_registration_images(
    images: list[Any],
    filenames: list[str],
) -> list[dict[str, Any]]:
    if not images:
        raise FacePipelineError("Debe enviar al menos una imagen.")

    if len(images) != len(filenames):
        raise FacePipelineError("Cada imagen debe tener un nombre de archivo.")

    samples = []

    for image, filename in zip(images, filenames):
        log(f"Procesando rostro de registro hibrido: {filename}")

        try:
            faces = detect_faces(image)
        except Exception as error:
            error_detail = str(error).splitlines()[0]
            raise FacePipelineError(
                f"RetinaFace no pudo procesar {filename}: {error_detail}"
            ) from error

        if not faces:
            raise FacePipelineError(
                f"No se detecto un rostro valido en {filename}."
            )

        if len(faces) > 1:
            raise FacePipelineError(
                f"Se detecto mas de un rostro en {filename}."
            )

        face = faces[0]
        embedding = generate_arcface_embedding(face["crop"])

        if embedding is None:
            raise FacePipelineError(
                f"ArcFace no pudo generar el embedding de {filename}."
            )

        try:
            normalized_embedding = l2_normalize_embedding(embedding)
        except ValueError as error:
            raise FacePipelineError(str(error)) from error

        samples.append(
            {
                "image": image,
                "filename": filename,
                "embedding": normalized_embedding,
                "quality": {
                    "bbox": face["bbox"],
                    "confidence": face["confidence"],
                    "sharpness": face["sharpness"],
                    "bbox_area_px": face["bbox_area_px"],
                    "landmarks": face["landmarks"],
                },
            }
        )

    return samples


def select_enrollment_embeddings(
    samples: list[dict[str, Any]],
    enrollment_strategy: str,
) -> list[list[float]]:
    embeddings = [sample["embedding"] for sample in samples]

    if enrollment_strategy == "multi":
        return embeddings

    try:
        return [calculate_embedding_centroid(embeddings)]
    except ValueError as error:
        raise FacePipelineError(str(error)) from error


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
        full_name=full_name,
        external_code=external_code,
        samples=samples,
        enrollment_strategy=enrollment_strategy,
        enrollment_embeddings=enrollment_embeddings,
    )
    result["pipeline"] = "hybrid"
    result["hybrid_analysis"] = analyses
    return result


def sign_in(
    *,
    image: Any,
    anomaly_detector: Any,
    threshold: float | None = None,
    match_count: int = 5,
) -> dict[str, Any]:
    log("Detectando y alineando rostro con RetinaFace...")

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

    face = faces[0]
    embedding = generate_arcface_embedding(face["crop"])

    if embedding is None:
        raise FacePipelineError("ArcFace no pudo generar el embedding.")

    try:
        normalized_embedding = l2_normalize_embedding(embedding)
    except ValueError as error:
        raise FacePipelineError(str(error)) from error

    analysis = analyze_embedding(normalized_embedding, anomaly_detector)
    reduced_embedding = np.array(analysis["umap_embedding"])
    cluster_id, probability, top_clusters = assign_cluster(reduced_embedding)
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

    effective_threshold = 0.40 if threshold is None else threshold

    if probability >= 0.5:
        result = match_face_embedding_by_cluster(
            normalized_embedding,
            cluster_id,
            match_count=match_count,
            threshold=effective_threshold,
        )
    else:
        result = None
        for cluster in top_clusters:
            candidate = match_face_embedding_by_cluster(
                normalized_embedding,
                cluster["cluster_id"],
                match_count=match_count,
                threshold=effective_threshold,
            )
            if (
                candidate["best_match"]
                and candidate["distance"] <= effective_threshold
            ):
                result = candidate
                break

        if result is None:
            result = match_face_embedding_by_cluster(
                normalized_embedding,
                -1,
                match_count=match_count,
                threshold=effective_threshold,
            )

    result["pipeline"] = "hybrid"
    result["quality"] = {
        "bbox": face["bbox"],
        "confidence": face["confidence"],
        "sharpness": face["sharpness"],
        "bbox_area_px": face["bbox_area_px"],
        "landmarks": face["landmarks"],
    }
    result["hybrid_analysis"] = {
        **analysis,
        "cluster_id": cluster_id,
        "cluster_probability": probability,
        "top_clusters": top_clusters,
    }

    return result
