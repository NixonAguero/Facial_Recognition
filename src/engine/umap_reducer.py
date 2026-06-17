import os
import joblib
import numpy as np
from scipy.spatial.distance import cosine as scipy_cosine

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "weights", "umap_faces.joblib")

from src.utils.constants import COMPONENTS_REDUCER, MIN_NEIGHBORS_REDUCER

_umap_model = None

def _get_model():
    global _umap_model
    if _umap_model is None:
        model = joblib.load(MODEL_PATH)

        # El modelo fue serializado con metric="cosine" en Colab (Python 3.10).
        # umap.transform() no usa self.metric directamente — usa self._input_distance_func,
        # que es pynn_named_distances["cosine"]: una función compilada con @numba.njit
        # de pynndescent. Ese bytecode numba es incompatible con Python 3.13.
        #
        # Solución: reemplazar _input_distance_func por scipy.cosine, que tiene
        # la misma firma (u, v) -> float y sklearn.pairwise_distances lo acepta
        # sin invocar numba.
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