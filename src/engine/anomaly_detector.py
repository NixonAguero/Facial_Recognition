"""
Detector de anomalías basado en el autoencoder facial.

Este módulo es la interfaz que consume el pipeline híbrido.
Encapsula: carga de pesos, inferencia, umbral de anomalía y decisión.

Flujo de uso:
    1. Al iniciar el sistema: AnomalyDetector.load(path)
    2. Durante verificación: detector.is_anomaly(embedding) → bool
    3. Si se necesita el score crudo: detector.score(embedding) → float

El umbral τ_anomaly se calibra por separado (ver calibrate_threshold.py)
y se almacena en threshold_config.yaml junto al umbral de similitud coseno.
Son umbrales independientes con propósitos distintos:
    - τ_similarity: separa misma-persona de distinta-persona (coseno)
    - τ_anomaly:    separa genuino de anomalía (MSE de reconstrucción)
"""

import os
import torch
import numpy as np
from src.engine.autoencoder import FacialAutoencoder


class AnomalyDetector:
    """
    Interfaz de inferencia del autoencoder para detección de anomalías.

    Atributos:
        model      (FacialAutoencoder): Red neuronal cargada.
        threshold  (float):            Umbral MSE sobre el cual se declara anomalía.
        device     (torch.device):     CPU o CUDA.
    """

    # Umbral por defecto conservador.
    # Debe reemplazarse con el valor calibrado por calibrate_threshold.py.
    DEFAULT_THRESHOLD = 0.05

    def __init__(self, model: FacialAutoencoder, threshold: float, device: torch.device):
        self.model     = model
        self.threshold = threshold
        self.device    = device
        self.model.eval()

    # ── Construcción ─────────────────────────────────────────────────────────

    @classmethod
    def load(cls, weights_path: str, threshold: float = None) -> "AnomalyDetector":
        """
        Carga el autoencoder desde disco y construye el detector.

        Args:
            weights_path: Ruta al archivo .pt generado por train_autoencoder.py.
            threshold:    Umbral MSE. Si es None, usa DEFAULT_THRESHOLD.
                          En producción debe venir de threshold_config.yaml.

        Returns:
            Instancia de AnomalyDetector lista para inferencia.

        Raises:
            FileNotFoundError: Si weights_path no existe.
        """
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"No se encontraron pesos del autoencoder en: {weights_path}\n"
                f"Ejecute scripts/train_autoencoder.py primero."
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = FacialAutoencoder().to(device)

        checkpoint = torch.load(weights_path, map_location=device, weights_only=True)

        # Soporte para guardar solo state_dict o el checkpoint completo
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            saved_threshold = checkpoint.get("threshold", cls.DEFAULT_THRESHOLD)
        else:
            # Compatibilidad: si se guardó directamente el state_dict
            model.load_state_dict(checkpoint)
            saved_threshold = cls.DEFAULT_THRESHOLD

        final_threshold = threshold if threshold is not None else saved_threshold
        return cls(model, final_threshold, device)

    # ── Inferencia ───────────────────────────────────────────────────────────

    def score(self, embedding: np.ndarray) -> float:
        """
        Calcula el score de anomalía (MSE de reconstrucción).

        Score alto  → embedding difícil de reconstruir → probable anomalía.
        Score bajo  → embedding dentro de la distribución aprendida → genuino.

        Args:
            embedding: Vector L2-normalizado de ArcFace. Shape: (512,).

        Returns:
            MSE escalar. Rango típico: [0.001, 0.2].
        """
        tensor = self._to_tensor(embedding)
        with torch.no_grad():
            mse = self.model.reconstruction_error(tensor)
        return float(mse.item())

    def is_anomaly(self, embedding: np.ndarray) -> bool:
        """
        Decisión binaria: ¿es este embedding una anomalía?

        Args:
            embedding: Vector L2-normalizado de ArcFace. Shape: (512,).

        Returns:
            True si score > threshold (anomalía detectada).
            False si score ≤ threshold (embedding genuino).
        """
        return self.score(embedding) > self.threshold

    def score_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Calcula scores de anomalía para un lote de embeddings.

        Más eficiente que llamar score() en un loop cuando se procesan
        múltiples imágenes simultáneamente (por ejemplo, durante el registro).

        Args:
            embeddings: Matriz de embeddings L2-normalizados. Shape: (N, 512).

        Returns:
            Array de MSE por muestra. Shape: (N,).
        """
        tensor = self._to_tensor(embeddings)
        with torch.no_grad():
            scores = self.model.reconstruction_error(tensor)
        return scores.cpu().numpy()

    # ── Utilidades ───────────────────────────────────────────────────────────

    def _to_tensor(self, embedding: np.ndarray) -> torch.Tensor:
        """
        Convierte numpy array a tensor en el device correcto.
        Maneja tanto embeddings individuales (512,) como lotes (N, 512).
        """
        arr = np.array(embedding, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]   # (512,) → (1, 512) para BatchNorm1d
        return torch.from_numpy(arr).to(self.device)

    def __repr__(self) -> str:
        return (
            f"AnomalyDetector("
            f"threshold={self.threshold:.4f}, "
            f"device={self.device}, "
            f"latent_dim={FacialAutoencoder.LATENT_DIM})"
        )