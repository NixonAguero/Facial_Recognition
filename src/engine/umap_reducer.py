import os
import joblib
import numpy as np
import umap

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "weights", "umap_faces.joblib")

def umap_reducer(embedding):
    umap_model = joblib.load(MODEL_PATH)
    
    input_data = np.array(embedding).reshape(1, -1)
    
    reduced = umap_model.transform(input_data)
    
    return reduced[0].tolist()

def train_and_save_umap(historical_embeddings):
    output_path = os.path.join(PROJECT_ROOT, "weights", "umap_faces.joblib")

    training_data = np.array(historical_embeddings)
    n_samples = training_data.shape[0]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    n_neighbors = min(15, n_samples - 1)
    n_components = 32
    
    init_method = "spectral" if n_samples > n_components + 1 else "random"
    metric = "cosine"

    print(f"UMAP params: n_samples={n_samples}, n_neighbors={n_neighbors}, n_components={n_components}, init={init_method}, metric={metric}")

    umap_model = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
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