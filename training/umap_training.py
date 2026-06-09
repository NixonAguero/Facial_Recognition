import os
import sys
import cv2
from ultralytics import YOLO

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
from src.engine.ArcFace import generate_arcface_embedding
from src.engine.umap_reducer import train_and_save_umap

def detect_and_crop_face(image_path, model_yolo):
    image_bgr = cv2.imread(image_path)

    if image_bgr is None:
        print(f"Error: Could not read image from path {image_path}")
        return None

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = model_yolo.predict(source=image_rgb, classes=[0], conf=0.5, verbose=False)
    matrix_boxes = results[0].boxes

    if len(matrix_boxes) == 0:
        print("No faces detected in the image.")
        return None

    box = matrix_boxes[0]
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    face_submatrix = image_rgb[y1:y2, x1:x2]

    if face_submatrix.size == 0:
        return None

    return face_submatrix


if __name__ == "__main__":
    yolo_model_path = os.path.join(PROJECT_ROOT, "weights", "yolov8n.pt")
    
    if not os.path.exists(yolo_model_path):
        print(f"Modelo YOLO no encontrado. Descargándolo automáticamente y moviéndolo a {yolo_model_path}...")
        os.makedirs(os.path.dirname(yolo_model_path), exist_ok=True)
        YOLO("yolov8n.pt")
        
        import shutil
        if os.path.exists("yolov8n.pt"):
            shutil.move("yolov8n.pt", yolo_model_path)
            
    yolo_model_instance = YOLO(yolo_model_path)

    sample_images = [
        "dataset/sign_in/notUser0.png",
        "dataset/sign_in/notUser1.png",
        "dataset/sign_in/notUser2.png",
        "dataset/sign_in/user0.png",
        "dataset/sign_in/user1.png",
        "dataset/sign_in/user2.png",
        "dataset/sign_in/user3.png"
    ]

    collected_embeddings = []

    for image in sample_images:
        abs_path = os.path.join(PROJECT_ROOT, image)
        face_matrix = detect_and_crop_face(abs_path, yolo_model_instance)

        if face_matrix is not None:
            embedding_vector = generate_arcface_embedding(face_matrix)

            if embedding_vector is not None:
                collected_embeddings.append(embedding_vector)
                print(f"Embedding generated successfully with {len(embedding_vector)} dimensions.")
            else:
                print(f"Failed to generate embedding for {image}")
        else:
            print(f"No face detected in {image}")

    print(f"\nTotal embeddings collected: {len(collected_embeddings)}")

    if len(collected_embeddings) >= 2:
        print("Training UMAP model...")
        train_and_save_umap(collected_embeddings)
    else:
        print("Error: Not enough valid embeddings to train UMAP (need at least 2).")