from pathlib import Path
from typing import Any

import cv2

from src.pipelines import hybrid_pipeline, standard_pipeline
from src.utils.constants import HYBRID
from src.utils.logger import log


def register_user(
    args: Any,
    anomaly_detector: Any | None = None,
) -> dict[str, Any]:
    raw_paths = (
        args.image_path
        if isinstance(args.image_path, list)
        else [args.image_path]
    )
    image_paths = [Path(path) for path in raw_paths]
    images = []

    for image_path in image_paths:
        log(f"Leyendo imagen de registro: {image_path}")
        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(f"No se pudo leer la imagen: {image_path}")

        images.append(image)

    if args.method == HYBRID:
        return hybrid_pipeline.sign_up(
            images=images,
            filenames=[path.name for path in image_paths],
            full_name=args.full_name,
            external_code=args.external_code,
            enrollment_strategy=args.enrollment_strategy,
            anomaly_detector=anomaly_detector,
        )

    return standard_pipeline.sign_up(
        images=images,
        filenames=[path.name for path in image_paths],
        full_name=args.full_name,
        external_code=args.external_code,
        enrollment_strategy=args.enrollment_strategy,
    )
