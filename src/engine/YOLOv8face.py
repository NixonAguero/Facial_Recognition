import cv2
import numpy as np
from ultralytics import YOLO


class YOLOv8Face:

    def __init__(
        self,
        model_path: str,
        conf: float = 0.25,
        iou: float = 0.45,
        min_confidence: float = 0.5,
        min_sharpness: float = 80.0,
        min_bbox_area: float = 6400.0,
        device: str = "",
    ):
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.min_confidence = min_confidence
        self.min_sharpness = min_sharpness
        self.min_bbox_area = min_bbox_area
        self.device = device

    def detect(self, image: np.ndarray) -> list[dict]:
        """
        Detecta rostros en una imagen BGR y aplica filtro de calidad.

        Returns:
            Lista de dicts con bbox, confidence, sharpness, landmarks.
            Solo rostros que pasaron el filtro.
        """
        results = self.model(
            image,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        has_keypoints = (
            result.keypoints is not None
            and result.keypoints.xy is not None
            and len(result.keypoints.xy) > 0
        )
        kpts = result.keypoints.xy.cpu().numpy() if has_keypoints else None

        detections = []
        for i in range(len(boxes_xyxy)):
            x1, y1, x2, y2 = boxes_xyxy[i]
            conf = float(confidences[i])
            area = float((x2 - x1) * (y2 - y1))

            if conf < self.min_confidence or area < self.min_bbox_area:
                continue

            crop = self._crop(image, x1, y1, x2, y2)
            sharpness  = self._sharpness(crop)
            brightness = self._brightness(crop)
            contrast   = self._contrast(crop)

            if sharpness < self.min_sharpness:
                continue
            if brightness < 40 or brightness > 220:
                continue
            if contrast < 20:
                continue

            landmarks = []
            if kpts is not None and i < len(kpts):
                landmarks = [
                    {"x": float(kx), "y": float(ky)}
                    for kx, ky in kpts[i]
                ]

            detections.append({
                "bbox": {
                    "x1": float(x1), "y1": float(y1),
                    "x2": float(x2), "y2": float(y2),
                    "width": float(x2 - x1),
                    "height": float(y2 - y1),
                },
                "confidence": conf,
                "sharpness": sharpness,
                "bbox_area_px": area,
                "landmarks": landmarks,
                "crop": crop,   # numpy BGR — listo para lo que siga
            })

        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections

    def warmup(self) -> None:
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy, verbose=False)

    def _crop(self, image, x1, y1, x2, y2) -> np.ndarray:
        h, w = image.shape[:2]
        return image[
            max(0, int(y1)):min(h, int(y2)),
            max(0, int(x1)):min(w, int(x2)),
        ]

    @staticmethod
    def _sharpness(crop: np.ndarray) -> float:
        if crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var()) 
        #Laplacian es el filtro que mide que tan borrosa es la imagen (BLUR)

    @staticmethod
    def _brightness(crop: np.ndarray) -> float:
        if crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return float(cv2.mean(gray)[0])

    @staticmethod
    def _contrast(crop: np.ndarray) -> float:
        if crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return float(gray.std())

    @staticmethod
    def decode_image(raw_bytes: bytes) -> np.ndarray:
        nparr = np.frombuffer(raw_bytes, dtype=np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("No se pudo decodificar la imagen.")
        return img