# Facial Recognition

Sistema de reconocimiento facial por línea de comandos con pipelines estándar
e híbrido.

## Arquitectura

```text
main -> api -> pipeline -> engine -> database
```

Los módulos principales son:

```text
src/
├── main.py
├── api/
│   ├── register.py
│   └── verify.py
├── pipelines/
│   ├── standard_pipeline.py
│   └── hybrid_pipeline.py
├── engine/
│   ├── RetinaFace.py
│   ├── ArcFace.py
│   ├── YOLOv8face.py
│   ├── autoencoder.py
│   ├── anomaly_detector.py
│   └── umap_reducer.py
├── database/
│   ├── face_repository.py
│   └── storage_repository.py
└── utils/
    ├── constants.py
    ├── logger.py
    └── normalizer.py
```

La carpeta `api` contiene adaptadores para la línea de comandos. No expone una
API web.

## Procesamiento facial

Los pipelines activos utilizan RetinaFace:

1. RetinaFace detecta los rostros.
2. Obtiene la caja, confianza y cinco landmarks.
3. Alinea el rostro utilizando los landmarks.
4. Normaliza los píxeles del crop al rango `[0, 1]`.
5. El filtro OpenCV valida sharpness, brillo, contraste y tamaño.
6. ArcFace ajusta el crop a `112x112` y genera 512 valores.
7. El pipeline aplica normalización L2 al embedding.

`YOLOv8face.py` se conserva en el proyecto, pero ya no es utilizado por los
pipelines estándar ni híbrido.

## Pipeline estándar

### Registro

El registro acepta una o varias imágenes y permite elegir una estrategia:

- `multi`: guarda un embedding por cada imagen.
- `centroid`: promedia los embeddings normalizados, vuelve a normalizar el
  resultado y guarda un único embedding representativo.

En ambos modos se guardan todas las imágenes procesadas.

### Verificación

1. Lee la imagen.
2. Ejecuta RetinaFace y ArcFace.
3. Consulta los embeddings almacenados mediante pgvector.
4. Devuelve la coincidencia y la distancia.

## Pipeline híbrido

Utiliza el mismo procesamiento de RetinaFace y ArcFace. Después ejecuta UMAP y
el detector de anomalías antes de registrar o buscar la identidad.

## Ejecución

Configure `SUPABASE_URL` y `SUPABASE_SECRET_KEY` e instale las dependencias.

Registro estándar:

```powershell
python -m src.main --action sign-up --method standard `
  --image-path dataset/test/ian1.png dataset/test/ian2.png `
    dataset/test/ian3.png dataset/test/ian4.png `
  --full-name "Test User" --external-code "USER-0"
```

Para guardar todos los embeddings:

```powershell
python -m src.main --action sign-up --method standard `
  --image-path dataset/test/ian1.png dataset/test/ian2.png `
    dataset/test/ian3.png dataset/test/ian4.png `
  --enrollment-strategy multi `
  --full-name "Ian" --external-code "Ian"
```

Para guardar el centroide:

```powershell
python -m src.main --action sign-up --method standard `
  --image-path dataset/test/ian1.png dataset/test/ian2.png `
    dataset/test/ian3.png dataset/test/ian4.png `
  --enrollment-strategy centroid `
  --full-name "Ian" --external-code "Ian"
```

Verificación estándar:

```powershell
python -m src.main --action sign-in --method standard `
  --image-path dataset/sign_in/user0.png
```

Para el pipeline híbrido, cambie `--method standard` por `--method hybrid`.

El `sign-in` continúa recibiendo una sola imagen. No necesita
`--enrollment-strategy`, porque la estrategia ya quedó representada en los
embeddings almacenados durante el registro.

## Calibración del threshold

El archivo `src/utils/calibrate_threshold.py` permite elegir entre:

- `far`: ROC con un FAR objetivo.
- `eer`: punto donde FAR y FRR son similares.
- `youden`: maximiza `TPR - FAR`.
- `cost`: minimiza el costo configurado de los errores.
- `percentile`: usa un percentil de las distancias genuinas.

### Generar las distancias

El formato recomendado organiza las fotos por persona:

```text
dataset/calibration/
├── Ian/
│   ├── foto1.png
│   ├── foto2.png
│   └── foto3.png
├── Nixon/
│   ├── foto1.png
│   └── foto2.png
└── Cristel/
    ├── foto1.png
    └── foto2.png
```

Para generar `data/distances.json`:

```powershell
python -m src.utils.generate_distances `
  --dataset-path dataset/calibration `
  --label-mode folders `
  --output-path data/distances.json
```

También se puede usar una carpeta plana con nombres como `ian1.png`,
`ian2.png`, `nixon1.png`:

```powershell
python -m src.utils.generate_distances `
  --dataset-path dataset/test `
  --label-mode filename-prefix `
  --output-path data/distances.json
```

El programa compara cada pareja de imágenes una sola vez:

- Misma persona: comparación genuina.
- Personas diferentes: comparación impostora.

Para una persona con `n` imágenes, crea `n × (n - 1) / 2` comparaciones
genuinas. Entre dos personas con `a` y `b` imágenes, crea `a × b`
comparaciones impostoras.

El archivo de entrada debe tener este formato:

```json
{
  "genuine_distances": [0.31, 0.42, 0.55],
  "impostor_distances": [0.81, 0.92, 1.03]
}
```

Método recomendado, FAR objetivo:

```powershell
python -m src.utils.calibrate_threshold `
  --input-path data/distances.json `
  --calibration-method far `
  --target-far 0.01 `
  --output-path config/threshold_result.json
```

Otros métodos:

```powershell
python -m src.utils.calibrate_threshold `
  --input-path data/distances.json `
  --calibration-method eer

python -m src.utils.calibrate_threshold `
  --input-path data/distances.json `
  --calibration-method youden

python -m src.utils.calibrate_threshold `
  --input-path data/distances.json `
  --calibration-method cost `
  --false-accept-cost 10 `
  --false-reject-cost 1

python -m src.utils.calibrate_threshold `
  --input-path data/distances.json `
  --calibration-method percentile `
  --percentile 95
```
