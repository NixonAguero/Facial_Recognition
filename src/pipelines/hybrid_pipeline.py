from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.pipelines.standard_pipeline import FacePipelineError
from src.engine.umap_reducer import umap_reducer
from src.utils.logger import log


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
    face = get_valid_face(image)
    embedding = generate_normalized_embedding(face)
    analysis = analyze_embedding(embedding, anomaly_detector)
    result = find_match(embedding, threshold, match_count)
    result["pipeline"] = "hybrid"
    result["quality"] = face_quality(face)
    result["hybrid_analysis"] = analysis
    return result
