import os
import joblib
import numpy as np
from scipy.spatial.distance import cosine as scipy_cosine
import umap
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "weights", "umap_faces.joblib")

from src.utils.constants import COMPONENTS_REDUCER, MIN_NEIGHBORS_REDUCER

_umap_model = None

def _get_model():
    global _umap_model
    if _umap_model is None:
        model = joblib.load(MODEL_PATH)

        model._input_distance_func = scipy_cosine
        model._metric_kwds = {}

        _umap_model = model
    return _umap_model

def umap_reducer(embedding):
    umap_model = _get_model()
    embedding_array = np.array(embedding, dtype=np.float32)
    input_data = embedding_array.reshape(1, -1)
    reduced = umap_model.transform(input_data)
    return reduced[0].tolist()

def train_and_save_umap(historical_embeddings):
    output_path = os.path.join(PROJECT_ROOT, "weights", "umap_faces.joblib")

    training_data = np.array(historical_embeddings)
    n_samples = training_data.shape[0]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    n_neighbors = min(MIN_NEIGHBORS_REDUCER, n_samples - 1)
    
    init_method = "spectral" if n_samples > COMPONENTS_REDUCER + 1 else "random"
    metric = "cosine"

    print(f"UMAP params: n_samples={n_samples}, n_neighbors={n_neighbors}, n_components={COMPONENTS_REDUCER}, init={init_method}, metric={metric}")

    umap_model = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=COMPONENTS_REDUCER,
        metric=metric,
        n_epochs=None,
        learning_rate=1.0,
        init=init_method,
        min_dist=0.1,
        spread=1.0,
        low_memory=False,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
        repulsion_strength=1.0,
        negative_sample_rate=5,
        transform_queue_size=4.0,
        a=None,
        b=None,
        random_state=42,
        angular_rp_forest=True,
        target_n_neighbors=-1,
        verbose=False
    )

    umap_model.fit(training_data)
    joblib.dump(umap_model, output_path)
    print(f"UMAP model saved at: {output_path}")

    print("UMAP Validation \n")
    vector_reduced = umap_reducer(historical_embeddings[0])
    print(f"UMAP results: \n embedding to reduce: \n{historical_embeddings[0]} \n new vector dimension {len(vector_reduced)} \n embedding reduced: \n {vector_reduced}")

    save_path = os.path.join(PROJECT_ROOT, "distribucion_umap.png")
    plot_umap_distribution(training_data, save_path=save_path)


def plot_umap_distribution(embeddings_512d: np.ndarray, save_path: str | None = None):
    """
    Carga el modelo UMAP entrenado, reduce los embeddings de 512d a 32d,
    luego los reduce a 2d para visualización y genera el gráfico.

    Args:
        embeddings_512d: Array de shape (n_samples, 512) — embeddings originales
                         de ArcFace usados para entrenar UMAP.
        save_path:       Ruta para guardar el gráfico. Si None, solo muestra.
    """
    umap_model = _get_model()

    print("Reduciendo 512d → 32d con modelo UMAP entrenado...")
    embeddings_32d = umap_model.transform(embeddings_512d)

    print("Reduciendo 32d → 2d para visualización...")
    reducer_2d = umap.UMAP(
        n_components=2,
        metric="euclidean",
        n_neighbors=30,
        min_dist=0.1,
        random_state=42,
        verbose=False,
    )
    embeddings_2d = reducer_2d.fit_transform(embeddings_32d)

    n_samples = len(embeddings_512d)

    plt.figure(figsize=(14, 10))
    plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        s=1,
        alpha=0.3,
        c="steelblue",
    )
    plt.title(f"Distribución UMAP — {n_samples} embeddings | {COMPONENTS_REDUCER}d → 2d")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Gráfico guardado en: {save_path}")

    plt.show()