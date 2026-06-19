import sys
import numpy as np
import hdbscan
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
HDBSCAN_PATH = PROJECT_ROOT / "weights" / "hdbscan_faces.joblib"

from src.utils.constants import (MIN_CLUSTER_SIZE, TOP_K_CLUSTERS)

def assign_cluster(embedding: np.ndarray) -> tuple[int, float, list[dict]]:
    """
    Recibe embedding reducido a 32d por UMAP.
    Retorna:
        cluster_id  → cluster más probable (-1 si es identidad nueva)
        probability → confianza de esa asignación
        top_clusters → lista de los TOP_K clusters más probables
                       para buscar en FAISS en caso de baja confianza
    """
    clusterer = joblib.load(HDBSCAN_PATH)
    vector = np.array(embedding).reshape(1, -1)

    # Cluster más probable
    labels, probabilities = hdbscan.approximate_predict(clusterer, vector)
    cluster_id = int(labels[0])
    probability = float(probabilities[0])

    #Probabilidad de pertenencia a todos los clusters
    soft = hdbscan.membership_vector(clusterer, vector)[0]
    #soft: array de shape (n_clusters,) con probabilidad por cluster

    # Fallback: si la probabilidad es baja, buscar en top K clusters
    top_indices = np.argsort(soft)[::-1][:TOP_K_CLUSTERS]
    top_clusters = [
        {"cluster_id": int(i), "probability": float(soft[i])}
        for i in top_indices
        if soft[i] > 0.0  # ignorar clusters con probabilidad 0
    ]

    return cluster_id, probability, top_clusters

def train_and_save_hdbscan(
    embeddings: np.ndarray,
    embedding_ids: list[str],
) -> dict | None:

    n_samples = len(embeddings)

    if n_samples < MIN_CLUSTER_SIZE:
        print(f"Error: Need at least {MIN_CLUSTER_SIZE} embeddings.")
        return None

    print("Fitting HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=10,
        min_samples=5,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
        allow_single_cluster=False,
    )

    labels = clusterer.fit_predict(embeddings)

    # Guardar modelo
    HDBSCAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clusterer, HDBSCAN_PATH)
    print(f"HDBSCAN model saved at: {HDBSCAN_PATH}")

    # Estadísticas
    n_clusters = len(set(labels) - {-1})
    n_noise = int(np.sum(labels == -1))

    print(f"\n{'='*50}")
    print(f"Embeddings processed : {n_samples}")
    print(f"Clusters found       : {n_clusters}")
    print(f"Noise points (-1)    : {n_noise} ({n_noise/n_samples*100:.1f}%)")
    print(f"{'='*50}")

    # Retornar mapeo id → cluster_id
    return {
        embedding_id: int(label)
        for embedding_id, label in zip(embedding_ids, labels)
    }