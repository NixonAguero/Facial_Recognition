"ESTE SCRIPT PRUEBA EL FLUJO DE INICIO DE SESIÓN CON RECONOCIMIENTO FACIAL, INCLUYENDO LA GENERACIÓN DEL EMBEDDING A PARTIR DE UNA IMAGEN DE INICIO DE SESIÓN Y LA BÚSQUEDA DE UNA COINCIDENCIA EN LA BASE DE DATOS."

from pathlib import Path
from utils.debug_prints import print_match_results
from database.face_repository import match_face_embedding
from engine.arcface_embedder import generate_embedding


def embedding_to_list(embedding):
    if hasattr(embedding, "tolist"):
        return embedding.tolist()

    return embedding


LOGIN_IMAGE = Path(__file__).parent.parent.parent / "dataset" / "sign_in" / "user0.png"

print("Generating login embedding...")
embedding = generate_embedding(str(LOGIN_IMAGE))
embedding = embedding_to_list(embedding)

print("Searching for match...")
result = match_face_embedding(
    embedding=embedding,
    match_count=5,
    threshold=0.40,
)

print_match_results(result)