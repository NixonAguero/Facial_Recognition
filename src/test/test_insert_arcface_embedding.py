from utils.supabase_client import supabase

from engine.arcface_embedder import generate_embedding

PROFILE_ID = "c278e83c-c1b8-460b-b1ee-79412e7f36df"
from pathlib import Path

IMAGE_PATH = Path(__file__).parent.parent.parent / "dataset" / "sign_up" / "user0.png"

embedding = generate_embedding(str(IMAGE_PATH))

if hasattr(embedding, "tolist"):
    embedding = embedding.tolist()

print("Embedding length:", len(embedding))

if len(embedding) != 512:
    raise ValueError(f"Expected 512 dimensions, got {len(embedding)}")

response = supabase.table("face_embeddings").insert({
    "profile_id": PROFILE_ID,
    "embedding": embedding,
    "model_name": "arcface",
    "model_version": "insightface-buffalo_l"
}).execute()

print(response.data)