import os
import torch
import numpy as np
from src.engine.autoencoder import FacialAutoencoder


class AnomalyDetector:

    DEFAULT_THRESHOLD = 0.5

    def __init__(self, model: FacialAutoencoder, threshold: float, device: torch.device):
        print(f"[AnomalyDetector] Inicializando detector...")
        print(f"[AnomalyDetector] Device: {device}")
        print(f"[AnomalyDetector] Threshold: {threshold}")
        self.model     = model
        self.threshold = threshold
        self.device    = device
        self.model.eval()
        print(f"[AnomalyDetector] Modelo en modo eval. Listo.")

    @classmethod
    def load(cls, weights_path: str, threshold: float = None) -> "AnomalyDetector":
        print(f"[AnomalyDetector.load] Cargando pesos desde: {weights_path}")

        if not os.path.isfile(weights_path):
            print(f"[AnomalyDetector.load] ERROR: No se encontro el archivo en {weights_path}")
            raise FileNotFoundError(
                f"No se encontraron pesos del autoencoder en: {weights_path}\n"
                f"Ejecute scripts/train_autoencoder.py primero."
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[AnomalyDetector.load] Device seleccionado: {device}")

        model = FacialAutoencoder().to(device)
        print(f"[AnomalyDetector.load] Arquitectura FacialAutoencoder creada.")

        checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
        print(f"[AnomalyDetector.load] Checkpoint cargado. Tipo: {type(checkpoint)}")

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            print(f"[AnomalyDetector.load] Checkpoint completo detectado.")
            model.load_state_dict(checkpoint["model_state_dict"])
            saved_threshold = checkpoint.get("threshold", cls.DEFAULT_THRESHOLD)
            print(f"[AnomalyDetector.load] Threshold guardado en checkpoint: {saved_threshold}")
        else:
            print(f"[AnomalyDetector.load] Solo state_dict detectado. Usando threshold por defecto.")
            model.load_state_dict(checkpoint)
            saved_threshold = cls.DEFAULT_THRESHOLD

        final_threshold = threshold if threshold is not None else saved_threshold
        print(f"[AnomalyDetector.load] Threshold final: {final_threshold}")
        print(f"[AnomalyDetector.load] Carga completada exitosamente.")

        return cls(model, final_threshold, device)

    def score(self, embedding: np.ndarray) -> float:
        print(f"[AnomalyDetector.score] Calculando MSE...")
        print(f"[AnomalyDetector.score] Shape del embedding: {np.array(embedding).shape}")

        tensor = self._to_tensor(embedding)
        print(f"[AnomalyDetector.score] Tensor shape: {tensor.shape} | Device: {tensor.device}")

        with torch.no_grad():
            mse = self.model.reconstruction_error(tensor)

        result = float(mse.item())
        print(f"[AnomalyDetector.score] MSE calculado: {result:.6f} | Threshold: {self.threshold:.6f}")

        return result

    def is_anomaly(self, embedding) -> bool:
        print(f"[AnomalyDetector.is_anomaly] Evaluando embedding...")

        embedding_array = np.array(embedding, dtype=np.float32)
        print(f"[AnomalyDetector.is_anomaly] Shape: {embedding_array.shape} | dtype: {embedding_array.dtype}")

        score = self.score(embedding_array)
        result = score > self.threshold

        print(f"[AnomalyDetector.is_anomaly] Score: {score:.6f} | Threshold: {self.threshold:.6f} | Es anomalia: {result}")

        return result

    def score_batch(self, embeddings: np.ndarray) -> np.ndarray:
        print(f"[AnomalyDetector.score_batch] Procesando batch...")
        print(f"[AnomalyDetector.score_batch] Shape del batch: {np.array(embeddings).shape}")

        tensor = self._to_tensor(embeddings)
        print(f"[AnomalyDetector.score_batch] Tensor shape: {tensor.shape} | Device: {tensor.device}")

        with torch.no_grad():
            scores = self.model.reconstruction_error(tensor)

        result = scores.cpu().numpy()
        print(f"[AnomalyDetector.score_batch] Scores calculados: {result}")
        print(f"[AnomalyDetector.score_batch] Min: {result.min():.6f} | Max: {result.max():.6f} | Media: {result.mean():.6f}")

        return result

    def _to_tensor(self, embedding: np.ndarray) -> torch.Tensor:
        print(f"[AnomalyDetector._to_tensor] Convirtiendo a tensor...")

        arr = np.array(embedding, dtype=np.float32)
        print(f"[AnomalyDetector._to_tensor] Shape original: {arr.shape}")

        if arr.ndim == 1:
            arr = arr[None, :]
            print(f"[AnomalyDetector._to_tensor] Shape expandido: {arr.shape}")

        tensor = torch.from_numpy(arr).to(self.device)
        print(f"[AnomalyDetector._to_tensor] Tensor listo: shape={tensor.shape} | device={tensor.device}")

        return tensor

    def __repr__(self) -> str:
        return (
            f"AnomalyDetector("
            f"threshold={self.threshold:.4f}, "
            f"device={self.device}, "
            f"latent_dim={FacialAutoencoder.LATENT_DIM})"
        )