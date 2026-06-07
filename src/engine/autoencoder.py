"""
Autoencoder para detección de anomalías en embeddings de ArcFace.

Opera sobre vectores de 512 dimensiones (salida normalizada L2 de ArcFace).
Se entrena únicamente con embeddings de rostros genuinos (usuarios registrados),
de modo que el error de reconstrucción (MSE) sea bajo para rostros conocidos
y alto para anomalías: spoofs, deepfakes, identidades fuera de distribución,
o embeddings de baja calidad que pasaron el filtro de calidad.

Arquitectura:
    Encoder: 512 → 256 → 128 → 64 (bottleneck)
    Decoder: 64  → 128 → 256 → 512

La activación LeakyReLU evita neuronas muertas en espacios de embedding densos.
BatchNorm estabiliza el entrenamiento sin necesidad de LR scheduling complejo.
La capa de salida usa Tanh porque los embeddings normalizados L2 de ArcFace
tienen valores en [-1, 1].
"""

import torch
import torch.nn as nn


class FacialAutoencoder(nn.Module):
    """
    Autoencoder simétrico sobre embeddings de ArcFace (512-d).

    Atributos:
        encoder (nn.Sequential): Reduce 512-d al espacio latente de 64-d.
        decoder (nn.Sequential): Reconstruye 512-d desde el espacio latente.
    """

    INPUT_DIM  = 512
    LATENT_DIM = 64

    def __init__(self):
        super(FacialAutoencoder, self).__init__()

        # ── Encoder ──────────────────────────────────────────────────────────
        self.encoder = nn.Sequential(
            # Capa 1: 512 → 256
            nn.Linear(self.INPUT_DIM, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),

            # Capa 2: 256 → 128
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),

            # Capa 3: 128 → 64 (bottleneck)
            nn.Linear(128, self.LATENT_DIM),
            nn.BatchNorm1d(self.LATENT_DIM),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        # ── Decoder ──────────────────────────────────────────────────────────
        self.decoder = nn.Sequential(
            # Capa 1: 64 → 128
            nn.Linear(self.LATENT_DIM, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),

            # Capa 2: 128 → 256
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),

            # Capa 3: 256 → 512 (reconstrucción)
            # Tanh porque los embeddings L2-normalizados tienen rango [-1, 1]
            nn.Linear(256, self.INPUT_DIM),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass completo: encode → decode.

        Args:
            x: Tensor de embeddings normalizados L2. Shape: (batch, 512).

        Returns:
            Tensor reconstruido. Shape: (batch, 512).
        """
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Solo el encoder. Útil para inspeccionar el espacio latente.

        Args:
            x: Tensor de embeddings normalizados L2. Shape: (batch, 512).

        Returns:
            Representación latente. Shape: (batch, 64).
        """
        return self.encoder(x)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calcula el MSE de reconstrucción por muestra.

        Este valor es el score de anomalía: alto = anomalía, bajo = genuino.

        Args:
            x: Tensor de embeddings normalizados L2. Shape: (batch, 512).

        Returns:
            MSE por muestra. Shape: (batch,).
        """
        x_hat = self.forward(x)
        # MSE por muestra (promedio sobre las 512 dimensiones)
        return torch.mean((x - x_hat) ** 2, dim=1)