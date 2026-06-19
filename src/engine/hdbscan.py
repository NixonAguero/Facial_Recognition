import sys
import numpy as np
import hdbscan
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
HDBSCAN_PATH = PROJECT_ROOT / "weights" / "hdbscan_faces.joblib"

from src.utils.constants import (MIN_CLUSTER_SIZE, TOP_K_CLUSTERS)

import umap
import matplotlib.pyplot as plt

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
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=5,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
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


def plot_hdbscan_clusters(embeddings: np.ndarray, save_path: str | None = None):
    """
    Carga el modelo HDBSCAN entrenado, extrae los labels conocidos
    y genera un gráfico 2D de la distribución de clusters.

    Args:
        embeddings: Array de shape (n_samples, 32) — los mismos embeddings
                    reducidos que se usaron para entrenar HDBSCAN.
        save_path:  Ruta para guardar el gráfico. Si None, solo muestra.
    """
    clusterer = joblib.load(HDBSCAN_PATH)
    labels = clusterer.labels_

    n_clusters = len(set(labels) - {-1})
    n_noise = int(np.sum(labels == -1))

    print(f"Clusters: {n_clusters} | Ruido: {n_noise} ({n_noise/len(labels)*100:.1f}%)")
    print("Reduciendo 32d → 2d para visualización...")

    reducer_2d = umap.UMAP(
        n_components=2,
        metric="euclidean",
        n_neighbors=30,
        min_dist=0.1,
        random_state=42,
        verbose=False,
    )
    embeddings_2d = reducer_2d.fit_transform(embeddings)

    mask_noise = labels == -1
    mask_clusters = labels != -1

    plt.figure(figsize=(14, 10))

    # Ruido en gris
    plt.scatter(
        embeddings_2d[mask_noise, 0],
        embeddings_2d[mask_noise, 1],
        s=1,
        alpha=0.2,
        c="lightgray",
        label=f"Ruido ({n_noise})",
    )

    # Clusters en color
    scatter = plt.scatter(
        embeddings_2d[mask_clusters, 0],
        embeddings_2d[mask_clusters, 1],
        s=2,
        alpha=0.6,
        c=labels[mask_clusters],
        cmap="tab20",
    )

    plt.colorbar(scatter, label="cluster_id")
    plt.title(f"HDBSCAN — {n_clusters} clusters | {n_noise} ruido | {len(labels)} total")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend(loc="upper right", markerscale=5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Gráfico guardado en: {save_path}")

    plt.show()

def print_hdbscan_stats():
    """
    Imprime estadísticas de distribución del modelo HDBSCAN entrenado.
    """
    clusterer = joblib.load(HDBSCAN_PATH)
    labels = clusterer.labels_

    total = len(labels)
    noise = int(np.sum(labels == -1))
    clustered = total - noise
    unique_clusters = sorted(set(labels) - {-1})
    n_clusters = len(unique_clusters)

    cluster_sizes = [int(np.sum(labels == c)) for c in unique_clusters]
    smallest = min(cluster_sizes) if cluster_sizes else 0
    largest = max(cluster_sizes) if cluster_sizes else 0
    average = float(np.mean(cluster_sizes)) if cluster_sizes else 0.0
    median = float(np.median(cluster_sizes)) if cluster_sizes else 0.0
    largest_pct = (largest / total * 100) if total > 0 else 0.0

    print(f"\n{'='*30}")
    print(f"HDBSCAN DISTRIBUTION ANALYSIS")
    print(f"{'='*30}")
    print(f"Total embeddings        : {total}")
    print(f"Clustered embeddings    : {clustered}")
    print(f"Noise embeddings (-1)   : {noise}")
    print(f"Clustered percentage    : {clustered/total*100:.2f}%")
    print(f"Noise percentage        : {noise/total*100:.2f}%")
    print(f"Number of clusters      : {n_clusters}")
    print(f"Smallest cluster size   : {smallest}")
    print(f"Largest cluster size    : {largest}")
    print(f"Average cluster size    : {average:.2f}")
    print(f"Median cluster size     : {median:.2f}")
    print(f"Largest cluster pct     : {largest_pct:.2f}%")
    print(f"{'='*30}\n")