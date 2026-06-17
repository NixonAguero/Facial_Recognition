MODEL_PATH = "models/yolov8n-face-lindevs.pt"
AUTOENCODER_MODEL_PATH = "models/autoencoder.pt"
DEVICE = ""         # "" = autodetectar CPU/CUDA/MPS

#Pipeline
HYBRID = 'hybrid'
STANDARD = 'standard'

# Filtro de calidad
MIN_CONFIDENCE = 0.4
MIN_SHARPNESS = 30.0


SINGIN = 'sign-in'
SINGUP = 'sign-up'

EMBEDDING_DIMENSION = 512

# HDBSCAN
MIN_CLUSTER_SIZE = 2
TOP_K_CLUSTERS = 3

# UMAP REDUCER
COMPONENTS_REDUCER = 32
MIN_NEIGHBORS_REDUCER = 15