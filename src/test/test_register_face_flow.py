
"ESTE SCRIPT PRUEBA EL FLUJO DE REGISTRO DE UNA NUEVA CARA, INCLUYENDO LA CREACIÓN DE PERFIL, SUBIDA DE IMAGEN, GENERACIÓN DE EMBEDDING Y GUARDADO EN LA BASE DE DATOS."

from pathlib import Path

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


PROFILE_NAME = "Mujer rubian"
EXTERNAL_CODE = "TEST_USER_3"

IMAGE_PATH = Path(__file__).parent.parent.parent / "dataset" / "sign_up" / "user3.png"

print("Getting or creating profile...")
profile = get_or_create_profile(
    full_name=PROFILE_NAME,
    external_code=EXTERNAL_CODE,
)

print("Profile:", profile["id"])

print("Uploading image to Supabase Storage...")
storage_path = upload_face_image(
    profile_id=profile["id"],
    image_path=str(IMAGE_PATH),
)

print("Image uploaded:", storage_path)

print("Saving image metadata...")
face_image = save_face_image(
    profile_id=profile["id"],
    image_path=storage_path,
    image_type="sign_up",
)

print("Face image saved:", face_image["id"])

print("Generating ArcFace embedding...")
embedding = generate_embedding(str(IMAGE_PATH))
embedding = embedding_to_list(embedding)

print("Embedding length:", len(embedding))

print("Saving face embedding...")
face_embedding = save_face_embedding(
    profile_id=profile["id"],
    image_id=face_image["id"],
    embedding=embedding,
)

print("Face embedding saved:", face_embedding["id"])