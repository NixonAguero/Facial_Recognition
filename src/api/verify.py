from fastapi import APIRouter, File, HTTPException, UploadFile, status
from src.engine.YOLOv8face import YOLOv8Face

router = APIRouter()
detector: YOLOv8Face | None = None

def set_detector(d: YOLOv8Face) -> None:
    global detector
    detector = d

def verify_user(args):
    print("Verifying user...")
    print(f"Method: {args.method}")

@router.post("/verify")
async def verify(file: UploadFile = File(...)):
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Archivo vacío.")

    image = YOLOv8Face.decode_image(raw_bytes)
    detections = detector.detect(image)

    return {
        "total_detected": len(detections),
        "faces": [
            {
                "bbox": d["bbox"],
                "confidence": d["confidence"],
                "sharpness": d["sharpness"],
                "landmarks": d["landmarks"],
            }
            for d in detections
        ],
    }