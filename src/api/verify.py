from pathlib import Path
from typing import Any

import cv2

from src.pipelines import hybrid_pipeline, standard_pipeline
from src.utils.logger import log

from src.utils.constants import HYBRID


def verify_user(
    args: Any,
    anomaly_detector: Any | None = None,
) -> dict[str, Any]:
    raw_path = (
        args.image_path[0]
        if isinstance(args.image_path, list)
        else args.image_path
    )
    image_path = Path(raw_path)
    log(f"Leyendo imagen de verificación: {image_path}")
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError("No se pudo leer la imagen.")

    if args.method == HYBRID:
        return hybrid_pipeline.sign_in(
            image=image,
            anomaly_detector=anomaly_detector,
        )

    return standard_pipeline.sign_in(
        image=image,
    )
