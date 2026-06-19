"""
Script de entrenamiento del autoencoder facial.

Uso:
    python scripts/train_autoencoder.py \
        --embeddings_path data/registered_embeddings.npy \
        --output_path     weights/autoencoder.pt \
        --epochs          100 \
        --batch_size      32 \
        --lr              1e-3 \
        --val_split       0.2

Flujo completo:
    1. Carga los embeddings de ArcFace de los usuarios registrados (npy).
    2. Los divide en train / val (estratificación no necesaria: todos son genuinos).
    3. Entrena el autoencoder minimizando MSE de reconstrucción.
    4. Usa Early Stopping sobre la perdida de validacion.
    5. Al terminar:
       - Guarda los pesos del mejor modelo en --output_path.
       - Calcula y guarda el umbral de anomalia tau sobre el conjunto de validacion
         como percentil 95 del MSE (cubre el 95% de los rostros genuinos).
       - Imprime un resumen del entrenamiento.

El umbral guardado en el checkpoint puede sobreescribirse con
calibrate_threshold.py si se dispone de pares negativos etiquetados.

Dependencias:
    torch, numpy, scikit-learn
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# Permite importar desde src/ sin instalar el paquete
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.engine.autoencoder import FacialAutoencoder


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena el autoencoder de deteccion de anomalias faciales."
    )
    parser.add_argument(
        "--embeddings_path", type=str, required=True,
        help="Ruta al .npy con embeddings L2-normalizados de ArcFace. Shape: (N, 512)."
    )
    parser.add_argument(
        "--output_path", type=str, default="weights/autoencoder.pt",
        help="Dónde guardar el checkpoint con pesos y umbral."
    )
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument(
        "--val_split", type=float, default=0.2,
        help="Fraccion del dataset usada para validacion y calibracion del umbral."
    )
    parser.add_argument(
        "--patience", type=int, default=15,
        help="Epocas sin mejora antes de Early Stopping."
    )
    parser.add_argument(
        "--threshold_percentile", type=float, default=95.0,
        help=(
            "Percentil del MSE de validacion usado como umbral tau. "
            "P95 cubre el 95%% de rostros genuinos como no-anomalia."
        )
    )
    return parser.parse_args()


# ── Entrenamiento ─────────────────────────────────────────────────────────────

def load_embeddings(path: str) -> torch.Tensor:
    """Carga y valida los embeddings desde un archivo .npy."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No se encontró el archivo de embeddings: {path}")

    embeddings = np.load(path).astype(np.float32)

    if embeddings.ndim != 2 or embeddings.shape[1] != FacialAutoencoder.INPUT_DIM:
        raise ValueError(
            f"Se esperaba shape (N, {FacialAutoencoder.INPUT_DIM}), "
            f"se obtuvo {embeddings.shape}"
        )

    print(f"  Embeddings cargados: {embeddings.shape[0]} muestras x {embeddings.shape[1]} dims")
    return torch.from_numpy(embeddings)


def build_dataloaders(
    embeddings: torch.Tensor,
    val_split: float,
    batch_size: int
) -> tuple[DataLoader, DataLoader]:
    """Divide en train/val y crea DataLoaders."""
    dataset   = TensorDataset(embeddings)
    n_val     = int(len(dataset) * val_split)
    n_train   = len(dataset) - n_val

    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, drop_last=False)

    print(f"  Train: {n_train} muestras | Val: {n_val} muestras")
    return train_loader, val_loader


def train_epoch(
    model: FacialAutoencoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """Una epoca de entrenamiento. Devuelve la perdida promedio."""
    model.train()
    total_loss = 0.0

    for (batch,) in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        reconstruction = model(batch)
        loss = criterion(reconstruction, batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch)

    return total_loss / len(loader.dataset)


def validate_epoch(
    model: FacialAutoencoder,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple[float, np.ndarray]:
    """
    Una epoca de validacion.

    Returns:
        (pérdida_promedio, array_de_mse_por_muestra)
        El array de MSE por muestra se usa para calibrar el umbral tau.
    """
    model.eval()
    total_loss  = 0.0
    all_mse     = []

    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            reconstruction = model(batch)
            loss = criterion(reconstruction, batch)
            total_loss += loss.item() * len(batch)

            # MSE por muestra para calibracion del umbral
            per_sample_mse = torch.mean((batch - reconstruction) ** 2, dim=1)
            all_mse.append(per_sample_mse.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    all_mse  = np.concatenate(all_mse)
    return avg_loss, all_mse


def calibrate_threshold(val_mse: np.ndarray, percentile: float) -> float:
    """
    Calcula el umbral tau como el percentil P del MSE sobre datos de validacion.

    Lógica: si el P95 del MSE de rostros genuinos es X, entonces cualquier
    embedding con MSE > X esta fuera de la distribucion de genuinos -> anomalia.

    Args:
        val_mse:    Array de MSE por muestra sobre el conjunto de validacion.
        percentile: Percentil deseado (típicamente 95.0).

    Returns:
        Umbral escalar tau.
    """
    tau = float(np.percentile(val_mse, percentile))
    coverage = float(np.mean(val_mse <= tau) * 100)
    print(f"\n  Calibracion del umbral tau:")
    print(f"    Percentil P{percentile:.0f} del MSE de validacion: {tau:.6f}")
    print(f"    Cobertura real de genuinos: {coverage:.1f}% (deberia ser ~{percentile:.0f}%)")
    return tau


def save_checkpoint(
    model: FacialAutoencoder,
    threshold: float,
    output_path: str,
    train_loss: float,
    val_loss: float,
    epoch: int
) -> None:
    """Guarda el estado del modelo y metadatos de entrenamiento."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "threshold":        threshold,
        "input_dim":        FacialAutoencoder.INPUT_DIM,
        "latent_dim":       FacialAutoencoder.LATENT_DIM,
        "best_epoch":       epoch,
        "train_loss":       train_loss,
        "val_loss":         val_loss,
    }, output_path)
    print(f"\n  Checkpoint guardado en: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  Entrenamiento del Autoencoder Facial")
    print(f"  Device: {device}")
    print(f"{'='*55}\n")

    # 1. Datos
    print("[1/4] Cargando embeddings...")
    embeddings                 = load_embeddings(args.embeddings_path)
    train_loader, val_loader   = build_dataloaders(embeddings, args.val_split, args.batch_size)

    # 2. Modelo, optimizador y criterio
    print("\n[2/4] Inicializando modelo...")
    model     = FacialAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parametros entrenables: {total_params:,}")

    # 3. Entrenamiento con Early Stopping
    print(f"\n[3/4] Entrenando por hasta {args.epochs} epocas (patience={args.patience})...")
    best_val_loss   = float("inf")
    best_val_mse    = None
    best_state      = None
    best_epoch      = 0
    epochs_no_imprv = 0

    for epoch in range(1, args.epochs + 1):
        train_loss          = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mse   = validate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        # Log cada 10 epocas o si hay mejora
        if epoch % 10 == 0 or val_loss < best_val_loss:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoca {epoch:4d}/{args.epochs} | "
                f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                f"LR: {lr_now:.2e}"
                + (" <- mejor" if val_loss < best_val_loss else "")
            )

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss   = val_loss
            best_val_mse    = val_mse.copy()
            best_state      = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch      = epoch
            epochs_no_imprv = 0
        else:
            epochs_no_imprv += 1
            if epochs_no_imprv >= args.patience:
                print(f"\n  Early Stopping en epoca {epoch} (sin mejora por {args.patience} epocas).")
                break

    # Restaurar el mejor estado
    model.load_state_dict(best_state)
    print(f"\n  Mejor epoca: {best_epoch} | Mejor val loss: {best_val_loss:.6f}")

    # 4. Calibracion del umbral y guardado
    print("\n[4/4] Calibrando umbral de anomalia y guardando checkpoint...")
    threshold = calibrate_threshold(best_val_mse, args.threshold_percentile)
    save_checkpoint(model, threshold, args.output_path, 0.0, best_val_loss, best_epoch)

    print(f"\n{'='*55}")
    print(f"  Entrenamiento completado.")
    print(f"  Umbral tau calibrado: {threshold:.6f}")
    print(f"  Para usar en inferencia:")
    print(f"    detector = AnomalyDetector.load('{args.output_path}')")
    print(f"    is_ok    = not detector.is_anomaly(embedding)")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
