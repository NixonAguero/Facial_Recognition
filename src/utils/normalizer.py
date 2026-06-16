from collections.abc import Sequence

import numpy as np
from src.utils.constants import EMBEDDING_DIMENSION


def l2_normalize_embedding(
    embedding: Sequence[float] | np.ndarray,
) -> list[float]:
    """Validate and L2-normalize an ArcFace embedding."""
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)

    if vector.size != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Expected {EMBEDDING_DIMENSION} dimensions, got {vector.size}"
        )

    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("ArcFace returned an invalid embedding.")

    return (vector / norm).tolist()


def calculate_embedding_centroid(
    embeddings: Sequence[Sequence[float]],
) -> list[float]:
    """Average L2-normalized embeddings and normalize the result."""

    if not embeddings:
        raise ValueError("At least one embedding is required.")

    normalized_embeddings = [
        l2_normalize_embedding(embedding)
        for embedding in embeddings
    ]
    centroid = np.mean(normalized_embeddings, axis=0)
    return l2_normalize_embedding(centroid)
