import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.engine.umap_reducer import train_and_save_umap
from src.database.face_repository import get_embeddings_registered

import numpy as np
import ast

def parse_embedding(emb):
    if isinstance(emb, str):
        return ast.literal_eval(emb)
    return emb

if __name__ == "__main__":
    collected_embeddings = get_embeddings_registered()

    embeddings_numeric = []
    for item in collected_embeddings:
        parsed_val = parse_embedding(item["embedding"])
        embeddings_numeric.append(parsed_val)

    embeddings_matrix = np.array(embeddings_numeric, dtype=np.float32)

    train_and_save_umap(embeddings_matrix)