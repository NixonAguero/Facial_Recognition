import cv2

def capture_frame():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("No se pudo abrir la camara")
        return None

    print("Presiona ESPACIO para capturar | ESC para cancelar")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("No se pudo leer el frame")
            break

        cv2.imshow("Camara - ESPACIO para capturar | ESC para cancelar", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:        # ESC — cancelar
            frame = None
            break
        elif key == 32:      # ESPACIO — capturar
            print("Imagen capturada")
            break

    cap.release()
    cv2.destroyAllWindows()

    return frame