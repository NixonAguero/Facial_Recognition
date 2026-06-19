import json
from pathlib import Path

from src.api import register, verify
from src.utils import args as cli_args
from src.utils.constants import AUTOENCODER_MODEL_PATH, HYBRID, SINGUP, THRESHOLD_ANOMALY
from src.engine.anomaly_detector import AnomalyDetector
from src.utils.logger import log

def run(args):
    anomaly_detector = None
    
    log("Cargando autoencoder...")
    if not Path(AUTOENCODER_MODEL_PATH).is_file():
        raise FileNotFoundError(
            f"No se encontro el autoencoder: {AUTOENCODER_MODEL_PATH}"
        )

    anomaly_detector = AnomalyDetector.load(AUTOENCODER_MODEL_PATH, threshold=THRESHOLD_ANOMALY)
    log("Autoencoder listo.")

    if args.action == SINGUP:
        return register.register_user(args, anomaly_detector)

    return verify.verify_user(args, anomaly_detector)


def main():
    args = cli_args.parse_args()
    log(f"Iniciando {args.action} con pipeline {args.method}.")

    try:
        result = run(args)
    except Exception as error:
        print(f"Error: {error}")
        raise SystemExit(1) from None

    log("Proceso finalizado correctamente.")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
