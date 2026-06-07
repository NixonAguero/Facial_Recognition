from database.face_repository import match_face_embedding
from engine.arcface_embedder import generate_embedding


def embedding_to_list(embedding):
    if hasattr(embedding, "tolist"):
        return embedding.tolist()

    return embedding


def identify_face(
    image_path: str,
    threshold: float = 0.40,
    match_count: int = 5,
) -> dict:
    """
    Complete face identification flow:
    1. Generate ArcFace embedding
    2. Search closest embeddings in Supabase
    3. Return match result
    """

    embedding = generate_embedding(image_path)
    embedding = embedding_to_list(embedding)

    result = match_face_embedding(
        embedding=embedding,
        match_count=match_count,
        threshold=threshold,
    )

    return result