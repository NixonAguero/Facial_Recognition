from fastapi import APIRouter, File, HTTPException, UploadFile, status
from src.engine.YOLOv8face import YOLOv8Face

router = APIRouter()
detector: YOLOv8Face | None = None


def set_detector(d: YOLOv8Face) -> None:
    global detector
    detector = d


def register_user(args):
    print("Registering user...")
    print(f"Method: {args.method}")

@router.post("/register")
async def register(file: UploadFile = File(...)):
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Archivo vacío.")

    image = YOLOv8Face.decode_image(raw_bytes)
    detections = detector.detect(image)

    if not detections:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="No se detectó ningún rostro válido.")

    if len(detections) > 1:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Se detectaron múltiples rostros. Envía solo uno.")

    face = detections[0]

    return {
        "bbox": face["bbox"],
        "confidence": face["confidence"],
        "sharpness": face["sharpness"],
        "landmarks": face["landmarks"],
    }