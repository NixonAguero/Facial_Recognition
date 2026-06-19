from pathlib import Path
from typing import Any

import cv2

from src.pipelines import hybrid_pipeline, standard_pipeline
from src.utils.constants import HYBRID
from src.utils.logger import log
from src.utils.capture_frame import capture_frame

def register_user(
    args: Any,
    anomaly_detector: Any | None = None,
) -> dict[str, Any]:

    image = capture_frame()

    if image is None:
        raise ValueError("No se pudo capturar la imagen de la cámara")

    return standard_pipeline.sign_up(
        images=[image],                        # lista con una sola imagen
        filenames=["camera_capture.jpg"],      # filename genérico
        full_name=args.full_name,
        external_code=args.external_code,
        enrollment_strategy=args.enrollment_strategy,
    )
