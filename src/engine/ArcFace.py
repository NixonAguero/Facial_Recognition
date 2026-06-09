from deepface import DeepFace

def generate_arcface_embedding(face_detected):

    try:

        # El face detected es la sub matriz de la imagen ingresada, tomando en cuenta solo la seccion de la cara que noto yolo
        # al final es una matriz de colores 

        # El enforce_detection en flase hace que no utilice sus algoritmos 
        # internos para detectar un rostro, ya que si esta activado 
        # y hay una variacion en el rostro y no se detecta al 100% ,
        # como un rostro alineado viendo al frente lanzara un error, 
        # pero si es una cara impresa si la acepta, el que este apagado
        #  solo le indica que saque el embedding de lo que tiene y punto

        # El detector_backend en false le indica al modelo que no busque rostros en la imagen
        # ya que esta fue detectada por yolo antes

        analysis_result = DeepFace.represent(
            img_path = face_detected,
            model_name = "ArcFace",
            enforce_detection = False, 
            detector_backend = "skip"
        )

        # Este es un ejemplo de lo que devolveria el modelo
        # [
        #     {
        #         "embedding": [0.0142, -0.0531, 0.1102, ..., 0.0089],  # Lista de 512 números flotantes (Norma L2 = 1)
        #         "facial_area": {"x": 0, "y": 0, "w": 112, "h": 112}, # El área analizada (el tamaño estandarizado)
        #         "face_confidence": 1.0                              # Certeza de detección (1.0 porque saltamos el detector)
        #     }
        # ]

        print(f"Embbeding generated: {analysis_result[0]["embedding"]}")
        return analysis_result[0]["embedding"]

    except Exception as e:
        print(f"Error generating the embedding {e}")
        return None