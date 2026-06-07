from database.face_repository import (
    get_or_create_profile,
    save_face_image,
    save_face_embedding,
)
from database.storage_repository import upload_face_image
from engine.arcface_embedder import generate_embedding


def embedding_to_list(embedding):
    if hasattr(embedding, "tolist"):
        return embedding.tolist()

    return embedding


def register_face(
    full_name: str,
    external_code: str,
    image_path: str,
    image_type: str = "sign_up",
) -> dict:
    """
    Complete face registration flow:
    1. Get or create profile
    2. Upload image to Supabase Storage
    3. Save image metadata
    4. Generate ArcFace embedding
    5. Save embedding linked to profile and image
    """

    profile = get_or_create_profile(
        full_name=full_name,
        external_code=external_code,
    )

    storage_path = upload_face_image(
        profile_id=profile["id"],
        image_path=image_path,
    )

    face_image = save_face_image(
        profile_id=profile["id"],
        image_path=storage_path,
        image_type=image_type,
    )

    embedding = generate_embedding(image_path)
    embedding = embedding_to_list(embedding)

    face_embedding = save_face_embedding(
        profile_id=profile["id"],
        image_id=face_image["id"],
        embedding=embedding,
    )

    return {
        "profile": profile,
        "face_image": face_image,
        "face_embedding": face_embedding,
    }