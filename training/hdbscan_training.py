import os
import sys
from pathlib import Path
import ast
import umap
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.database.face_repository import (get_embeddings_registered, save_clusters)
from src.engine.umap_reducer import umap_reducer
from src.engine.hdbscan import (train_and_save_hdbscan, plot_hdbscan_clusters, print_hdbscan_stats)
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

def plot_raw_embeddings(embeddings_512d: np.ndarray, save_path: str | None = None):
    """
    Reduce los embeddings originales de 512d directamente a 2d con UMAP
    para visualizar su distribución antes de cualquier procesamiento.

    Args:
        embeddings_512d: Array de shape (n_samples, 512).
        save_path:       Ruta para guardar el gráfico. Si None, solo muestra.
    """
    n_samples = len(embeddings_512d)
    print(f"Reduciendo {n_samples} embeddings de 512d → 2d para visualización...")

    reducer_2d = umap.UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=30,
        min_dist=0.1,
        random_state=42,
        verbose=False,
    )
    embeddings_2d = reducer_2d.fit_transform(embeddings_512d)

    plt.figure(figsize=(14, 10))
    plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        s=1,
        alpha=0.3,
        c="steelblue",
    )
    plt.title(f"Embeddings crudos desde DB — {n_samples} muestras (512d → 2d)")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Gráfico guardado en: {save_path}")

    plt.show()

def plot_reduced_embeddings(embeddings_32d: np.ndarray, save_path: str | None = None):
    """
    Visualiza los embeddings ya reducidos a 32d por UMAP,
    reduciéndolos a 2d para el gráfico.

    Args:
        embeddings_32d: Array de shape (n_samples, 32).
        save_path:      Ruta para guardar el gráfico. Si None, solo muestra.
    """
    n_samples = len(embeddings_32d)
    print(f"Reduciendo {n_samples} embeddings de 32d → 2d para visualización...")

    reducer_2d = umap.UMAP(
        n_components=2,
        metric="euclidean",
        n_neighbors=30,
        min_dist=0.1,
        random_state=42,
        verbose=False,
    )
    embeddings_2d = reducer_2d.fit_transform(embeddings_32d)

    plt.figure(figsize=(14, 10))
    plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        s=1,
        alpha=0.3,
        c="darkorange",
    )
    plt.title(f"Embeddings reducidos por UMAP — {n_samples} muestras (32d → 2d)")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Gráfico guardado en: {save_path}")

    plt.show()

if __name__ == "__main__":

    embeddings_registered = get_embeddings_registered()

    if not embeddings_registered:
        sys.exit(1) 

    embedding_ids = [row["id"] for row in embeddings_registered]

    embeddings_to_arrays = []
    for embedding in embeddings_registered:
        parsed = parse_embedding(embedding["embedding"])
        embeddings_to_arrays.append(np.array(parsed))

    plot_raw_embeddings(
        embeddings_to_arrays,
        save_path=os.path.join(PROJECT_ROOT, "raw_embeddings.png")
    )

    embeddings_reduced = embedding_reducer(embeddings_to_arrays)
    embeddings_reduced = np.array(embeddings_reduced)

    plot_reduced_embeddings(
        embeddings_reduced,
        save_path=os.path.join(PROJECT_ROOT, "distribucion_umap.png")
    )

    cluster_map = train_and_save_hdbscan(embeddings_reduced, embedding_ids) 

    if not cluster_map:
        print("HDBSCAN training failed.")
        sys.exit(1)

    print(f"cluster map {cluster_map}")

    print_hdbscan_stats()

    plot_hdbscan_clusters(
        embeddings_reduced,
        save_path=os.path.join(PROJECT_ROOT, "clusters_hdbscan.png")
    )

    print("Updating cluster_id in database...")
    save_clusters(cluster_map)