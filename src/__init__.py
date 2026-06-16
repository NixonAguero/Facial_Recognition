import os


# RetinaFace requires the legacy Keras API with TensorFlow 2.16 or newer.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
