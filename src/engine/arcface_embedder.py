from pathlib import Path
import cv2
import numpy as np
from insightface.app import FaceAnalysis


# Load ArcFace / InsightFace model once when this file is imported
# ctx_id=-1 means CPU
# ctx_id=0 means GPU, if you have CUDA configured
_app = FaceAnalysis(name="buffalo_l")
_app.prepare(ctx_id=-1)


def generate_embedding(image_path: str) -> np.ndarray:
    """
    Generates a 512-dimensional face embedding using InsightFace / ArcFace.

    Args:
        image_path: Path to the image file.

    Returns:
        np.ndarray: 512-dimensional embedding.

    Raises:
        FileNotFoundError: If image does not exist.
        ValueError: If image cannot be read or no face is detected.
    """

    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    faces = _app.get(image)

    if len(faces) == 0:
        raise ValueError("No face detected in the image.")

    # If there are multiple faces, use the largest one
    face = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    )

    embedding = face.embedding

    if embedding is None:
        raise ValueError("Face detected, but embedding could not be generated.")

    return embedding