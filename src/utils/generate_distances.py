import argparse
import json
import re
from collections import Counter
from collections.abc import Callable
from itertools import combinations
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.engine.ArcFace import generate_arcface_embedding
from src.engine.RetinaFace import detect_faces
from src.utils.logger import log
from src.utils.normalizer import l2_normalize_embedding


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
LABEL_MODES = ("auto", "folders", "filename-prefix")


def generate_image_embedding(image_path: Path) -> list[float]:
    """Generate one normalized embedding with the active face flow."""

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("No se pudo leer la imagen.")

    faces = detect_faces(image)
    if not faces:
        raise ValueError("No se detectó un rostro válido.")
    if len(faces) > 1:
        raise ValueError("Se detectó más de un rostro.")

    embedding = generate_arcface_embedding(faces[0]["crop"])
    if embedding is None:
        raise ValueError("ArcFace no pudo generar el embedding.")

    return l2_normalize_embedding(embedding)


def discover_labeled_images(
    dataset_path: str | Path,
    label_mode: str = "auto",
) -> tuple[list[dict[str, Any]], str]:
    """Find images and assign an identity to each one."""

    root = Path(dataset_path)
    if not root.is_dir():
        raise ValueError(f"No existe el dataset: {root}")
    if label_mode not in LABEL_MODES:
        raise ValueError(
            f"Modo de etiquetas inválido: {label_mode}"
        )

    selected_mode = label_mode
    if selected_mode == "auto":
        has_identity_folders = any(
            child.is_dir()
            and any(is_image_file(path) for path in child.rglob("*"))
            for child in root.iterdir()
        )
        selected_mode = (
            "folders"
            if has_identity_folders
            else "filename-prefix"
        )

    if selected_mode == "folders":
        samples = discover_from_folders(root)
    else:
        samples = discover_from_filename_prefix(root)

    if not samples:
        raise ValueError("No se encontraron imágenes en el dataset.")

    return samples, selected_mode


def discover_from_folders(root: Path) -> list[dict[str, Any]]:
    samples = []

    for identity_folder in sorted(root.iterdir()):
        if not identity_folder.is_dir():
            continue

        for image_path in sorted(identity_folder.rglob("*")):
            if is_image_file(image_path):
                samples.append(
                    {
                        "identity": identity_folder.name,
                        "path": image_path,
                    }
                )

    return samples


def discover_from_filename_prefix(root: Path) -> list[dict[str, Any]]:
    samples = []

    for image_path in sorted(root.iterdir()):
        if not is_image_file(image_path):
            continue

        samples.append(
            {
                "identity": extract_identity_from_filename(
                    image_path.stem
                ),
                "path": image_path,
            }
        )

    return samples


def extract_identity_from_filename(filename: str) -> str:
    """Convert names such as ian1 or ian_02 into identity ian."""

    identity = re.sub(r"[\s_-]*\d+$", "", filename).strip()
    if not identity:
        raise ValueError(
            f"No se pudo obtener la identidad desde: {filename}"
        )
    return identity


def is_image_file(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def process_images(
    samples: list[dict[str, Any]],
    dataset_path: str | Path,
    embedding_generator: Callable[[Path], list[float]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Generate embeddings and report images that cannot be processed."""

    root = Path(dataset_path)
    generator = embedding_generator or generate_image_embedding
    processed = []
    skipped = []

    for sample in samples:
        image_path = sample["path"]
        relative_path = str(image_path.relative_to(root))
        log(f"Procesando imagen de calibración: {relative_path}")

        try:
            embedding = generator(image_path)
        except Exception as error:
            skipped.append(
                {
                    "identity": sample["identity"],
                    "image": relative_path,
                    "reason": str(error),
                }
            )
            continue

        processed.append(
            {
                "identity": sample["identity"],
                "image": relative_path,
                "embedding": embedding,
            }
        )

    return processed, skipped


def create_distance_pairs(
    samples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compare every unordered pair exactly once."""

    genuine_pairs = []
    impostor_pairs = []

    for first, second in combinations(samples, 2):
        distance = cosine_distance(
            first["embedding"],
            second["embedding"],
        )
        comparison = {
            "identity_a": first["identity"],
            "image_a": first["image"],
            "identity_b": second["identity"],
            "image_b": second["image"],
            "distance": distance,
        }

        if first["identity"] == second["identity"]:
            genuine_pairs.append(comparison)
        else:
            impostor_pairs.append(comparison)

    return genuine_pairs, impostor_pairs


def cosine_distance(
    first_embedding: list[float],
    second_embedding: list[float],
) -> float:
    first = np.asarray(
        l2_normalize_embedding(first_embedding),
        dtype=np.float64,
    )
    second = np.asarray(
        l2_normalize_embedding(second_embedding),
        dtype=np.float64,
    )
    similarity = float(np.clip(np.dot(first, second), -1.0, 1.0))
    return 1.0 - similarity


def generate_distances(
    dataset_path: str | Path,
    output_path: str | Path | None = None,
    label_mode: str = "auto",
    embedding_generator: Callable[[Path], list[float]] | None = None,
) -> dict[str, Any]:
    """Generate genuine and impostor distances from a labeled dataset."""

    discovered, selected_mode = discover_labeled_images(
        dataset_path,
        label_mode,
    )
    processed, skipped = process_images(
        discovered,
        dataset_path,
        embedding_generator,
    )
    identity_counts = Counter(
        sample["identity"] for sample in processed
    )

    if len(identity_counts) < 2:
        raise ValueError(
            "Se necesitan al menos dos personas con imágenes válidas."
        )
    if not any(count >= 2 for count in identity_counts.values()):
        raise ValueError(
            "Al menos una persona debe tener dos imágenes válidas."
        )

    genuine_pairs, impostor_pairs = create_distance_pairs(processed)
    if not genuine_pairs:
        raise ValueError("No se pudieron crear comparaciones genuinas.")
    if not impostor_pairs:
        raise ValueError("No se pudieron crear comparaciones impostoras.")

    result = {
        "metric": "cosine_distance",
        "label_mode": selected_mode,
        "dataset_path": str(Path(dataset_path)),
        "identity_image_counts": dict(sorted(identity_counts.items())),
        "processed_images": len(processed),
        "skipped_images": skipped,
        "genuine_pair_count": len(genuine_pairs),
        "impostor_pair_count": len(impostor_pairs),
        "genuine_distances": [
            pair["distance"] for pair in genuine_pairs
        ],
        "impostor_distances": [
            pair["distance"] for pair in impostor_pairs
        ],
        "genuine_comparisons": genuine_pairs,
        "impostor_comparisons": impostor_pairs,
    }

    if output_path is not None:
        save_distance_file(result, output_path)

    return result


def save_distance_file(
    result: dict[str, Any],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate genuine and impostor face distances."
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Dataset organized by folders or filename prefixes.",
    )
    parser.add_argument(
        "--output-path",
        default="data/distances.json",
        help="JSON file where distances are saved.",
    )
    parser.add_argument(
        "--label-mode",
        choices=LABEL_MODES,
        default="auto",
        help="How image identities are obtained.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = generate_distances(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        label_mode=args.label_mode,
    )

    print(
        json.dumps(
            {
                "output_path": args.output_path,
                "label_mode": result["label_mode"],
                "identity_image_counts": result[
                    "identity_image_counts"
                ],
                "processed_images": result["processed_images"],
                "skipped_images": len(result["skipped_images"]),
                "genuine_pair_count": result[
                    "genuine_pair_count"
                ],
                "impostor_pair_count": result[
                    "impostor_pair_count"
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
