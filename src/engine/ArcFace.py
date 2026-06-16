from deepface import DeepFace


def generate_arcface_embedding(normalized_face):
    try:
        # RetinaFace already detected, aligned and normalized the face.
        analysis_result = DeepFace.represent(
            img_path=normalized_face,
            model_name="ArcFace",
            enforce_detection=False,
            detector_backend="skip",
        )

        return analysis_result[0]["embedding"]

    except Exception as error:
        print(f"Error generando el embedding: {error}")
        return None
