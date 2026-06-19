from typing import Any
import time
from src.utils.supabase_client import supabase


def create_profile(full_name: str, external_code: str | None = None) -> dict[str, Any]:
    """
    Creates a new person/profile in Supabase.
    """

    data = {
        "full_name": full_name,
        "external_code": external_code,
    }

    response = supabase.table("profiles").insert(data).execute()

    if not response.data:
        raise RuntimeError("Profile could not be created.")

    return response.data[0]


def save_face_embedding(
    profile_id: str,
    embedding: list[float],
    image_id: str | None = None,
    model_name: str = "arcface",
    model_version: str = "insightface-buffalo_l",
) -> dict[str, Any]:
    """
    Saves a face embedding linked to a profile.
    """

    if len(embedding) != 512:
        raise ValueError(f"Expected 512 dimensions, got {len(embedding)}")

    data = {
        "profile_id": profile_id,
        "image_id": image_id,
        "embedding": embedding,
        "model_name": model_name,
        "model_version": model_version,
    }

    response = supabase.table("face_embeddings").insert(data).execute()

    if not response.data:
        raise RuntimeError("Face embedding could not be saved.")

    return response.data[0]


def match_face_embedding(
    embedding: list[float],
    match_count: int = 5,
    threshold: float = 0.50,
) -> dict[str, Any]:
    """
    Searches for the closest face embedding in Supabase.
    Returns match result with accepted/rejected decision.
    """

    if len(embedding) != 512:
        raise ValueError(f"Expected 512 dimensions, got {len(embedding)}")

    response = supabase.rpc("match_face_embedding", {
        "query_embedding": embedding,
        "match_count": match_count,
    }).execute()

    matches = response.data or []

    if not matches:
        return {
            "matched": False,
            "reason": "No candidates found.",
            "best_match": None,
            "matches": [],
        }

    best_match = matches[0]
    distance = best_match["distance"]

    return {
        "matched": distance <= threshold,
        "distance": distance,
        "threshold": threshold,
        "best_match": best_match,
        "matches": matches,
    }


def match_face_embedding_by_cluster(
    embedding: list[float],
    cluster_id: int,
    match_count: int = 5,
    threshold: float = 0.50,
) -> dict[str, Any]:

    if len(embedding) != 512:
        raise ValueError(f"Expected 512 dimensions, got {len(embedding)}")

    response = supabase.rpc("match_face_embedding_by_cluster", {
        "query_embedding": embedding,
        "match_cluster_id": cluster_id,
        "match_count": match_count,
    }).execute()

    matches = response.data or []

    if not matches:
        return {
            "matched": False,
            "reason": "No candidates found.",
            "best_match": None,
            "matches": [],
        }

    best_match = matches[0]
    distance = best_match["distance"]

    return {
        "matched": distance <= threshold,
        "distance": distance,
        "threshold": threshold,
        "best_match": best_match,
        "matches": matches,
    }


def get_or_create_profile(full_name: str, external_code: str) -> dict:
    """
    Gets an existing profile by external_code.
    If it does not exist, creates it.
    """

    existing_profile = get_profile_by_external_code(external_code)

    if existing_profile:
        return existing_profile

    return create_profile(
        full_name=full_name,
        external_code=external_code
    )



def get_profile_by_external_code(external_code: str):
    """
    Finds a profile by external_code.
    Returns the profile if it exists, otherwise None.
    """

    response = (
        supabase
        .table("profiles")
        .select("*")
        .eq("external_code", external_code)
        .limit(1)
        .execute()
    )

    if response.data:
        return response.data[0]

    return None

def get_embeddings_registered():
    all_embeddings = []
    page_size = 1000
    offset = 0

    while True:
        response = (
            supabase
            .table("face_embeddings")
            .select("id, embedding")
            .eq("is_active", True)
            .range(offset, offset + page_size - 1)
            .execute()
        )

        if not response.data:
            break

        all_embeddings.extend(response.data)

        if len(response.data) < page_size:
            break

        offset += page_size

    return all_embeddings if all_embeddings else None

def save_clusters(cluster_map: dict):
    BATCH_SIZE = 500
    items = list(cluster_map.items())
    total = len(items)

    print(f"Guardando {total} clusters en {(total / BATCH_SIZE)} lotes...")

    for i in range(0, total, BATCH_SIZE):
        batch = items[i:i + BATCH_SIZE]

        payload = [
            {"id": eid, "cluster_id": cid}
            for eid, cid in batch
        ]

        supabase.rpc("bulk_update_clusters", {"payload": payload}).execute()

        print(f"Progreso: {min(i + BATCH_SIZE, total)}/{total}")

    print("Clusters guardados correctamente.")

def save_face_image(
    profile_id: str,
    image_path: str,
    image_type: str | None = None,
) -> dict:
    """
    Saves image metadata in the face_images table.
    """

    data = {
        "profile_id": profile_id,
        "image_path": image_path,
        "image_type": image_type,
    }

    response = supabase.table("face_images").insert(data).execute()

    if not response.data:
        raise RuntimeError("Face image metadata could not be saved.")

    return response.data[0]
