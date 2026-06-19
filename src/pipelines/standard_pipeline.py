from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import cv2

from src.database.face_repository import (
    get_or_create_profile,
    match_face_embedding,
    save_face_embedding,
    save_face_image,
)
from src.database.storage_repository import upload_face_image
from src.engine.ArcFace import generate_arcface_embedding
from src.engine.RetinaFace import detect_faces
from src.utils.logger import log
from src.utils.normalizer import (
    calculate_embedding_centroid,
    l2_normalize_embedding,
)
from src.utils.constants import (MIN_HEIGHT, MIN_WIDTH)

class FacePipelineError(RuntimeError):
    pass

def save_registration(
    *,
    full_name: str,
    external_code: str,
    samples: list[dict[str, Any]],
    enrollment_strategy: str,
    enrollment_embeddings: list[list[float]],
) -> dict[str, Any]:
    """Save the profile, images and embeddings produced during sign-up."""

    log("Guardando perfil, imagenes y embeddings...")
    profile = get_or_create_profile(
        full_name=full_name,
        external_code=external_code,
    )

    face_images = []

    for sample in samples:
        suffix = Path(sample["filename"]).suffix or ".jpg"

        with NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        try:
            image_was_written = cv2.imwrite(
                str(temporary_path),
                sample["image"],
            )

            if not image_was_written:
                raise FacePipelineError(
                    f"No se pudo preparar {sample['filename']} para guardarla."
                )

            storage_path = upload_face_image(
                profile_id=profile["id"],
                image_path=str(temporary_path),
            )
        finally:
            temporary_path.unlink(missing_ok=True)

        face_image = save_face_image(
            profile_id=profile["id"],
            image_path=storage_path,
            image_type="sign_up",
        )
        face_images.append(face_image)

    face_embeddings = []

    if enrollment_strategy == "multi":
        for face_image, embedding in zip(
            face_images,
            enrollment_embeddings,
        ):
            face_embedding = save_face_embedding(
                profile_id=profile["id"],
                image_id=face_image["id"],
                embedding=embedding,
                model_name="ArcFace",
                model_version="deepface-multi",
            )
            face_embedding.pop("embedding", None)
            face_embeddings.append(face_embedding)
    else:
        face_embedding = save_face_embedding(
            profile_id=profile["id"],
            image_id=None,
            embedding=enrollment_embeddings[0],
            model_name="ArcFace",
            model_version="deepface-centroid",
        )
        face_embedding.pop("embedding", None)
        face_embeddings.append(face_embedding)

    return {
        "profile": profile,
        "face_images": face_images,
        "face_embeddings": face_embeddings,
        "enrollment_strategy": enrollment_strategy,
        "processed_images": len(samples),
        "stored_embeddings": len(face_embeddings),
        "pipeline": "standard",
        "quality": [sample["quality"] for sample in samples],
    }


def sign_up(
    *,
    images: list[Any],
    filenames: list[str],
    full_name: str,
    external_code: str,
    enrollment_strategy: str,
) -> dict[str, Any]:
    """Register one person using one or more face images."""

    if not images:
        raise FacePipelineError("Debe enviar al menos una imagen.")

    if len(images) != len(filenames):
        raise FacePipelineError(
            "Cada imagen debe tener un nombre de archivo."
        )

    samples = []

    for image, filename in zip(images, filenames):
        log(f"Procesando rostro de registro: {filename}")

        try:
            faces = detect_faces(image)
        except Exception as error:
            error_detail = str(error).splitlines()[0]
            raise FacePipelineError(
                f"RetinaFace no pudo procesar {filename}: {error_detail}"
            ) from error

        if not faces:
            raise FacePipelineError(
                f"No se detecto un rostro valido en {filename}."
            )

        if len(faces) > 1:
            raise FacePipelineError(
                f"Se detecto mas de un rostro en {filename}."
            )

        face = faces[0]

        crop = face["crop"]

        height, weight = crop.shape[:2]
        if height < MIN_HEIGHT and weight < MIN_WIDTH:
            print("Se detecto un recorte con dimensiones mínimas a las permitidas para generar un embedding")
            failed_detection += 1
            return None

        embedding = generate_arcface_embedding(face["crop"])

        if embedding is None:
            raise FacePipelineError(
                f"ArcFace no pudo generar el embedding de {filename}."
            )

        try:
            normalized_embedding = l2_normalize_embedding(embedding)
        except ValueError as error:
            raise FacePipelineError(str(error)) from error

        samples.append(
            {
                "image": image,
                "filename": filename,
                "embedding": normalized_embedding,
                "quality": {
                    "bbox": face["bbox"],
                    "confidence": face["confidence"],
                    "sharpness": face["sharpness"],
                    "bbox_area_px": face["bbox_area_px"],
                    "landmarks": face["landmarks"],
                },
            }
        )

    embeddings = [sample["embedding"] for sample in samples]

    if enrollment_strategy == "multi":
        enrollment_embeddings = embeddings
    else:
        try:
            centroid = calculate_embedding_centroid(embeddings)
        except ValueError as error:
            raise FacePipelineError(str(error)) from error

        enrollment_embeddings = [centroid]

    return save_registration(
        full_name=full_name,
        external_code=external_code,
        samples=samples,
        enrollment_strategy=enrollment_strategy,
        enrollment_embeddings=enrollment_embeddings,
    )


def sign_in(
    *,
    image: Any,
    threshold: float | None = None,
    match_count: int = 5,
) -> dict[str, Any]:
    """Identify a person from one face image."""

    log("Detectando y alineando rostro con RetinaFace...")

    try:
        faces = detect_faces(image)
    except Exception as error:
        error_detail = str(error).splitlines()[0]
        raise FacePipelineError(
            f"RetinaFace no pudo procesar la imagen: {error_detail}"
        ) from error

    if not faces:
        raise FacePipelineError("No se detecto un rostro valido.")

    if len(faces) > 1:
        raise FacePipelineError("Se detecto mas de un rostro.")

    face = faces[0]
    crop = face["crop"]

    height, weight = crop.shape[:2]
    if height < MIN_HEIGHT and weight < MIN_WIDTH:
        print("Se detecto un recorte con dimensiones mínimas a las permitidas para generar un embedding")
        failed_detection += 1
        return None

    embedding = generate_arcface_embedding(face["crop"])

    if embedding is None:
        raise FacePipelineError("ArcFace no pudo generar el embedding.")

    try:
        normalized_embedding = l2_normalize_embedding(embedding)
    except ValueError as error:
        raise FacePipelineError(str(error)) from error

    log("Buscando coincidencias en la base de datos...")

    if threshold is None:
        result = match_face_embedding(
            embedding=normalized_embedding,
            match_count=match_count,
        )
    else:
        result = match_face_embedding(
            embedding=normalized_embedding,
            match_count=match_count,
            threshold=threshold,
        )

    result["pipeline"] = "standard"
    result["quality"] = {
        "bbox": face["bbox"],
        "confidence": face["confidence"],
        "sharpness": face["sharpness"],
        "bbox_area_px": face["bbox_area_px"],
        "landmarks": face["landmarks"],
    }

    return result
