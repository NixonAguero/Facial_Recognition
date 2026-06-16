import os
from pathlib import Path
from uuid import uuid4

from src.utils.supabase_client import supabase

BUCKET_NAME = os.getenv("SUPABASE_BUCKET", "face-images")
# Parámetros:
    #   profile_id: ID del usuario (ej: "c278e83c")
    #   image_path: Ruta local de la imagen (ej: "dataset/sign_up/user0.png")
    # Retorna: Ruta guardada en Supabase (ej: "profiles/c278e83c/abc123.png")



def upload_face_image(profile_id: str, image_path: str) -> str:
     """
    Uploads a face image to Supabase Storage and returns the storage path.
    """
     path = Path(image_path) # Convierte string a objeto Path

     if not path.exists():
           raise FileNotFoundError(f"Image not found: {image_path}")
     
     file_extension = path.suffix.lower() 
        # Si image_path = "user0.png" → file_extension = ".png"
        # Si image_path = "user0.JPG" → file_extension = ".jpg"

     storage_path = f"profiles/{profile_id}/{uuid4()}{file_extension}"
        # Ejemplo:
        # profile_id = "c278e83c"
        # uuid4() = "550e8400-e29b-41d4-a716-446655440000"
        # Resultado: "profiles/c278e83c/550e8400-e29b-41d4-a716-446655440000.png"
        # (Genera ID único para que no haya conflictos)


     with open(path, "rb") as file: # "rb" = read binary (leer como bytes)
           supabase.storage.from_(BUCKET_NAME).upload(
                 path=storage_path, # Dónde guardarlo
                 file=file,  # Contenido del archivo
                 file_options={
                       "content_type": _get_content_type(file_extension),
                       "upsert": "false",  # No sobrescribir si existe
                 },
           )
     return storage_path # Devuelve: "profiles/c278e83c/550e8400...png"

def _get_content_type(file_extension: str) -> str:
     # Le dice a Supabase qué tipo de archivo es
    if file_extension in [".jpg", ".jpeg"]:
        return "image/jpeg"

    if file_extension == ".png":
        return "image/png"

    return "application/octet-stream"# Tipo genérico/desconocido



