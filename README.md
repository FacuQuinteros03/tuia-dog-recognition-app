# TP2 - Sistema de Detección y Clasificación de Razas de Perros

Plantilla base para desarrollar un pipeline completo de Computer Vision:

**Embeddings → Búsqueda por similitud → Clasificación → Detección → Pipeline completo**

> 📄 El enunciado completo del trabajo práctico está en [ENUNCIADO.md](ENUNCIADO.md) (versión PDF: [Trabajo Práctico N°2.pdf](<Trabajo Práctico N°2.pdf>)).

La cátedra provee la infraestructura completa (Docker, aplicación Gradio, base de datos vectorial, API, scripts y herramientas de evaluación y visualización). El grupo hace un **fork** de este repositorio, trabaja en una **feature branch** y completa **únicamente** las funciones indicadas en cada etapa. Cualquier modificación fuera de esas funciones debe justificarse en el Pull Request.

## Objetivo del backend

API asincrónica en Python que permite:

- Búsqueda por similitud (`/search`): imagen consultada, Top-K imágenes similares y raza predicha.
- Clasificación supervisada (`/classify`): raza y score con el modelo entrenado seleccionado.
- Detección + clasificación (`/detect`): bounding boxes, raza y scores de confianza.
- Consulta del estado del procesamiento asincrónico (`/status/{job_id}`).
- Selección dinámica del modelo de embeddings (`/models` + campo `model` en `/search`).

La API responde `HTTP 202` con un `job_id` y luego permite consultar el resultado con estado:

```json
{
  "status": "done | inProgress | failed",
  "link": "url | none"
}
```

## Estructura

```text
tuia-dog-recognition-app/
├── 1-pre-processing.ipynb       # Parte 1: dataset y preprocesamiento
├── 2-train.ipynb                # Etapa 2: entrenamiento, evaluación (incl. NDCG@10 de la Etapa 1) y comparación
├── ENUNCIADO.md                 # Enunciado completo del TP
├── Trabajo Práctico N°2.pdf     # Enunciado en PDF
├── src/
│   ├── app/
│   │   └── main.py              # App FastAPI + /health
│   ├── frontend/                # UI Gradio provista por la cátedra (una pestaña por etapa)
│   │   ├── app.py
│   │   └── gradio_ui.py
│   └── lib/
│       ├── api.py               # Endpoints /upload, /search, /classify, /detect, /status, /models, /files
│       ├── bootstrap.py         # Construcción de los servicios (compartida por la API y los scripts)
│       ├── config.py            # Configuración vía .env
│       ├── files.py
│       ├── schemas.py
│       ├── evaluation/
│       │   └── metrics.py       # NDCG@k, precision/recall/F1, specificity
│       ├── visualization/
│       │   └── draw.py          # Dibujo de bounding boxes
│       ├── services/
│       │   ├── similarity_service.py   # ⚠️ Etapa 1: funciones a implementar
│       │   ├── classifier_service.py   # ⚠️ Etapa 2: funciones a implementar
│       │   ├── detection_service.py    # ⚠️ Etapa 3: funciones a implementar
│       │   └── task_manager.py
│       └── storage/
│           ├── base.py
│           ├── embedding_store.py      # Base vectorial en JSON
│           └── pgvector_store.py       # Base vectorial en PostgreSQL + pgvector
├── scripts/
│   ├── download_dataset.py      # Descarga el dataset de Kaggle
│   ├── build_index.py           # Indexa el dataset en la base vectorial
│   └── train_classifier.py      # Entrena/evalúa el clasificador (alternativa local a 2-train.ipynb)
├── data/
│   ├── dataset/                 # 70 Dog Breeds Image Dataset, no se versiona
│   │   └── clean/               # Dataset filtrado y re-particionado (1-pre-processing.ipynb); es el DATASET_PATH
│   └── embeddings.json          # Base vectorial JSON (si USE_PGVECTOR=false)
├── models/                      # Checkpoints entrenados (.pth / .onnx), no se versionan
├── output/                      # Imágenes subidas y resultados, no se versiona
├── requirements.txt
├── Dockerfile
├── Dockerfile.frontend
├── docker-compose.yml
├── .env.docker.example
└── .env.local.example
```

# Preparando el ambiente

## Requisitos

- Python 3.12
- Docker (Docker Desktop incluye Docker Compose)
- Git
- Cuenta de Kaggle (para descargar el dataset)

## Dataset

Dataset principal: [70 Dog Breeds Image Dataset (Kaggle)](https://www.kaggle.com/datasets/gpiosenka/70-dog-breedsimage-data-set).

```bash
python scripts/download_dataset.py
```

Requiere [credenciales de Kaggle](https://www.kaggle.com/docs/api) configuradas. Si no las tienen, descarguen el dataset manualmente desde Kaggle y descompriman `train/`, `valid/` y `test/` dentro de `data/dataset`.

Después hay que ejecutar `1-pre-processing.ipynb`: filtra el dataset (duplicados, imágenes borrosas o sin perro), lo re-particiona 70/15/15 y genera `data/dataset/clean/`, que es el `DATASET_PATH` que usa la app.

### Conjunto independiente de evaluación

133 imágenes de Wikimedia Commons, revisadas a mano: 100 fotos de un perro de 24 razas del dataset (Etapas 1 y 2), 13 escenas con varios perros y 20 sin perros (Etapa 3). Las imágenes no se versionan. La fuente, el autor y la licencia de cada una están en [`data/independent_sources.csv`](data/independent_sources.csv).

- **Descarga (solo lectura):** [conjunto_independiente.zip en Google Drive](https://drive.google.com/drive/folders/1SCEwhbLpWVE176pZPthhyTmz7qaeHBnc)
- Descomprimir `conjunto_independiente.zip` dentro de `data/dataset/`. Crea `independent/<raza>/` e `independent_scenes/{multi_dog,no_dog}/`. El zip trae también `ATRIBUCIONES.md` con el autor y la licencia de cada imagen.
- Alternativa sin el link: `python scripts/build_independent_set.py download` las baja de Commons a partir del CSV.

## Configurá tus modelos

Descargá los checkpoints entrenados desde el link de [Modelos y entrenamiento](#modelos-y-entrenamiento) (`resnet18_finetuned.pth` y `cnn_custom.pth`) y copialos a la carpeta `models`, o reentrenalos con `2-train.ipynb` (`RETRAIN = True`). El backend admite modelos con extensión **.pth** (PyTorch, se carga con `torch.load` el modelo completo) o **.onnx** (se carga con `onnxruntime`). Si usan otro framework, expórtenlo a **.onnx**.

Recordá actualizar la configuración del `.env` correspondiente (`RESNET18_MODEL_NAME`, `CNN_CUSTOM_MODEL_NAME`, `EMBEDDING_MODEL`, `EMBEDDING_DIM`, `SIMILARITY_THRESHOLD`) para que apunte a tus modelos.

> El backend arranca aunque falten los checkpoints: `/health` informa cuáles encuentra en `models/`. Las pestañas de las Etapas 2 y 3 fallan hasta que existan.

Con docker compose, el entorno reinicia la aplicación y actualiza el código automáticamente al detectar cambios. Lo mismo ocurre en local si usan `--reload`.

El reinicio automático puede no funcionar en algunas versiones de Docker Desktop para *Windows*. En ese caso, vuelvan a correr los comandos del siguiente apartado para actualizar el código dentro de Docker.

## Opción 1 - Corriendo dentro de Docker

Es la forma en que la cátedra evalúa el trabajo.

### 1. Construí y corré la aplicación

Ajustá el archivo **.env.docker.example** a tus necesidades. Luego corré desde la terminal:

```bash
docker compose build
docker compose up -d
```

Servicios disponibles: frontend en `http://localhost:8080`, backend en `http://localhost:8000` y PostgreSQL en `localhost:5432`.

### 2. Indexá la galería

Con la Etapa 1 implementada y el dataset en `data/dataset`, indexá el split de entrenamiento desde el contenedor del backend:

```bash
docker compose exec backend python /scripts/build_index.py --split train
```

## Opción 2 - Configurando el ambiente local

### 1. Cloná tu fork

```bash
git clone <url-de-tu-fork>
cd tuia-dog-recognition-app
git checkout -b feature/tp2-razas-perros
```

### 2. Instalá uv

Para usuarios de Linux o Mac:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Para usuarios de Windows:

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

[Documentación de uv](https://docs.astral.sh/uv/getting-started/installation/). También pueden usar `python3.12 -m venv .venv` y `pip`.

### 3. Configurá un ambiente virtual con Python 3.12

```bash
uv venv --python 3.12 .venv
```

### 4. Activá el ambiente virtual

```bash
# Linux / Mac
source .venv/bin/activate
```

```powershell
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 5. Instalá las dependencias

```bash
uv pip install -r requirements.txt
```

### 6. Configurá las variables de entorno

Ajustá el archivo `.env.local.example` a tus necesidades y copialo a `src/.env`. El backend local y los scripts de `scripts/` leen la configuración desde ese archivo.

```bash
cp .env.local.example src/.env
```

### 7. Descargá el dataset

```bash
python scripts/download_dataset.py
```

### 8. Iniciá la base de datos

```bash
docker compose up postgres -d
```

Solo es necesaria con `USE_PGVECTOR=true` (por defecto). Con `USE_PGVECTOR=false` se usa la base vectorial JSON (`data/embeddings.json`) y no hace falta Docker para trabajar en local.

### 9. Iniciá el backend

```bash
cd src
uvicorn app.main:app --reload --port 8000
```

### 10. Iniciá el frontend (en otra terminal, con el ambiente virtual activado)

```bash
cd src
uvicorn frontend.app:app --port 8080
```

### 11. Verificá que todo esté corriendo

- Frontend: `http://localhost:8080`
- Backend: `http://localhost:8000/docs` (Swagger) y `http://localhost:8000/health` (muestra el modelo seleccionado y los checkpoints encontrados)

## Scripts provistos

Se ejecutan desde la raíz del repositorio con el ambiente local activo (y PostgreSQL corriendo si `USE_PGVECTOR=true`). Leen la configuración de `src/.env` si existe:

```bash
# Descargar el dataset de Kaggle a data/dataset
python scripts/download_dataset.py

# Indexar la galería en la base vectorial (requiere la Etapa 1 implementada)
python scripts/build_index.py --split train
# ... o lo mismo desde el contenedor del backend (con docker compose up -d corriendo)
docker compose exec backend python /scripts/build_index.py --split train

# Bajar el conjunto independiente desde Wikimedia Commons (o armar el zip para compartirlo)
python scripts/build_independent_set.py download
python scripts/build_independent_set.py package

# Entrenar y evaluar el clasificador localmente (alternativa a 2-train.ipynb)
python scripts/train_classifier.py --model resnet18_finetuned
```

Con `USE_PGVECTOR=false`, la indexación del split completo es lenta: el JSON se reescribe en cada inserción. Para pruebas rápidas, usen `--limit` (máximo de imágenes por raza).

## Configuración

No hardcodeen parámetros: configúrenlos mediante `.env`.

1. Para la ejecución local, copien `.env.local.example` a `src/.env`.
2. Ajusten el modelo de embeddings (`EMBEDDING_MODEL`, `EMBEDDING_DIM`), la búsqueda (`SIMILARITY_METRIC`, `SIMILARITY_THRESHOLD`, `TOP_K`), los checkpoints (`RESNET18_MODEL_NAME`, `CNN_CUSTOM_MODEL_NAME`), YOLO (`YOLO_MODEL`, `YOLO_CONF_THRESHOLD`, `YOLO_DOG_CLASS_ID`) y los paths.
3. Opcional (habilitado por defecto): configuren la conexión a PostgreSQL + pgvector (`USE_PGVECTOR`). Con `USE_PGVECTOR=false`, los embeddings se guardan en un archivo JSON (`EMBEDDINGS_PATH`).

## Endpoints

- Backend: `http://localhost:8000` (documentación Swagger en `http://localhost:8000/docs`)
- PostgreSQL/pgvector: `localhost:5432`
- Frontend (provisto por la cátedra): `http://localhost:8080`

| Endpoint | Descripción |
|----------|-------------|
| `POST /upload` | Sube una imagen al servidor |
| `POST /search` | Etapa 1: búsqueda por similitud (acepta `model` y `top_k`) |
| `POST /classify` | Etapa 2: clasificación con el modelo entrenado (acepta `model`) |
| `POST /detect` | Etapa 3: detección + clasificación |
| `GET /status/{job_id}` | Estado del procesamiento asincrónico |
| `GET /models` | Modelos de embeddings disponibles |
| `GET /health` | Estado del backend y checkpoints encontrados |

## Pipeline

### Provisto por la cátedra

- API asincrónica (`/upload`, `/search`, `/classify`, `/detect`, `/status`, `/models`) y manejo de jobs.
- Orquestación de cada etapa: `search` e `index_image` (Etapa 1), `classify_image` (Etapa 2) y `predict` (Etapa 3: detección → recorte → clasificación → JSON).
- Selección dinámica del modelo de embeddings: `baseline`, `resnet18_finetuned` o `cnn_custom`.
- Carga de checkpoints (`.pth` / `.onnx`) desde `models/` y selección del modelo activo.
- Similitud configurable (`cosine` o `l2`) y manejo de desconocidos con `SIMILARITY_THRESHOLD`.
- Persistencia configurable en JSON o en PostgreSQL + pgvector (`USE_PGVECTOR`).
- Funciones auxiliares de evaluación (`lib/evaluation/metrics.py`) y de visualización (`lib/visualization/draw.py`).
- Frontend Gradio con una pestaña por etapa: Etapa 1 (imagen consultada, Top-K similares, raza predicha y selección de modelo), Etapa 2 (raza y score con el modelo entrenado) y Etapa 3 (bounding boxes, razas y scores).

### Implementado por el grupo

| Etapa | Función | Implementación |
|-------|---------|----------------|
| 1 | `extract_embedding` | ResNet18 de ImageNet sin `fc` (512), preprocesamiento compartido, normalización L2; el modelo se carga una vez |
| 1 | `search_similar_images` | pgvector (`<=>`) o recorrido del JSON; score con `self.similarity` (`cosine` o `l2`, mismo ranking con embeddings normalizados) |
| 1 | `predict_breed_from_neighbors` | Voto ponderado por similitud; score = proporción del voto de la raza ganadora; `unknown` si el vecino más cercano queda bajo `SIMILARITY_THRESHOLD` |
| 2 | `train_classifier` | Modelo A o B según el modelo activo; AdamW, *warmup* + coseno, *label smoothing*, *early stopping* por accuracy de valid; guarda el modelo completo con `classes` y el historial por época |
| 2 | `evaluate_classifier` | Accuracy, precision / recall / F1 macro y especificidad one-vs-rest promediada, sobre `test` |
| 2 | `extract_custom_embedding` | Capa previa al clasificador del modelo activo, normalizada L2 |
| 3 | `detect_dogs` | YOLOv8n de COCO, filtrado por `YOLO_DOG_CLASS_ID` y `YOLO_CONF_THRESHOLD`; lista vacía si no hay perros |
| 3 | `classify_detected_dog` | Modelo activo de la Etapa 2 sobre el recorte (letterbox, softmax) |

Módulos agregados: `src/lib/preprocessing.py` (preprocesamiento y augmentation compartidos),
`src/lib/models/` (arquitecturas, dataset de PyTorch e hiperparámetros) y el helper
`ClassifierService.predict_breed`, que usan la pestaña de la Etapa 2 y la Etapa 3.

Contrato original de cada función:

| Etapa | Función | Archivo |
|-------|---------|---------|
| 1 | `extract_embedding(image)` | `src/lib/services/similarity_service.py` |
| 1 | `search_similar_images(embedding, top_k)` | `src/lib/services/similarity_service.py` |
| 1 | `predict_breed_from_neighbors(results)` | `src/lib/services/similarity_service.py` |
| 2 | `train_classifier()` | `src/lib/services/classifier_service.py` |
| 2 | `evaluate_classifier()` | `src/lib/services/classifier_service.py` |
| 2 | `extract_custom_embedding(image)` | `src/lib/services/classifier_service.py` |
| 3 | `detect_dogs(image)` | `src/lib/services/detection_service.py` |
| 3 | `classify_detected_dog(crop)` | `src/lib/services/detection_service.py` |

En las Etapas 1 y 3 el trabajo se limita a completar las funciones. En la Etapa 2, además, el grupo entrena sus propios modelos (Modelo A: fine-tuning de ResNet18/ResNet50; Modelo B: CNN propia, **ambos obligatorios**) y hace un estudio de resultados.

Ver el contrato de cada función, recomendaciones y errores frecuentes en [ENUNCIADO.md](ENUNCIADO.md#24-funciones-a-implementar).

### Orden de trabajo sugerido

1. `1-pre-processing.ipynb`: EDA, filtrado, conjunto independiente, preprocesamiento, augmentation y splits.
2. Etapa 1 en `similarity_service.py`. Indexá la galería con `python scripts/build_index.py --split train` y probá la pestaña de la Etapa 1.
3. Etapa 2 en `classifier_service.py`, entrenando y comparando los modelos con `2-train.ipynb`. Descargá los checkpoints a `models/`.
4. Etapa 3 en `detection_service.py` y validación end-to-end con `docker compose up`.

## Notebooks

| Notebook | Contenido |
|----------|-----------|
| `1-pre-processing.ipynb` | Dataset, EDA, control de calidad, conjunto independiente, preprocesamiento, data augmentation y splits |
| `2-train.ipynb` | Evaluación de la búsqueda (NDCG@10), entrenamiento del Modelo A y del Modelo B, métricas, análisis de errores, comparación y exportación de checkpoints |

Se pueden entregar commiteadas en el repositorio o en Google Colab, con el link a la notebook de Colab en una celda Markdown al inicio de la notebook del repositorio. En ambos casos deben estar **ejecutadas y con las salidas visibles**: una notebook sin salidas se considera **no entregada** (ver [ENUNCIADO.md](ENUNCIADO.md#82-notebooks-repositorio-o-google-colab)).

## Dataset

Resumen de `1-pre-processing.ipynb`:

| Conjunto | Razas | Imágenes | Por raza |
|---|---|---|---|
| Dataset original (Kaggle) | 70 | 9346 | 10–198 |
| Descartadas por el filtrado | – | 782 | – |
| `train` / `valid` / `test` (70/15/15 estratificado, semilla 42) | 70 | 5994 / 1285 / 1285 | 53–143 / 11–31 / 11–31 |
| Independiente: fotos de un perro | 24 | 100 | 2–5 |
| Independiente: varios perros / sin perros | – | 13 / 20 | – |

- **Filtrado** (reglas con umbrales justificados en la notebook): imágenes borrosas (5), duplicadas
  o casi duplicadas por hash perceptual o embedding (774: 906 pares, 319 de ellos entre splits
  distintos del dataset original, y 28 con la misma foto bajo dos razas) y sin perro (3). Después se re-particiona sin
  fuga entre splits y el resultado queda en `data/dataset/clean/` (el `DATASET_PATH` de la app).
- **Conjunto independiente:** Wikimedia Commons, licencias libres, autor y licencia por imagen en
  `data/independent_sources.csv` (ver [Conjunto independiente de evaluación](#conjunto-independiente-de-evaluación)).
- **Preprocesamiento** (`src/lib/preprocessing.py`, el mismo en entrenamiento, búsqueda y
  detección): BGR → RGB, *letterbox* a 224 × 224 (sin deformar los recortes alargados de YOLO),
  normalización de ImageNet.
- **Data augmentation** (solo en entrenamiento): recorte aleatorio, flip horizontal, rotación y
  escala leves, brillo y contraste, desenfoque y ruido. Sin flips verticales ni cambios de tono
  que alteren el pelaje.

## Modelos y entrenamiento

| | Modelo | Detalle |
|---|---|---|
| Etapa 1 (baseline) | ResNet18 de ImageNet sin `fc`, embedding de 512 normalizado L2 | Misma dimensión (`EMBEDDING_DIM=512`) y arquitectura que el Modelo A |
| Etapa 2, Modelo A | ResNet18 *fine-tuned* (`src/lib/models/resnet_classifier.py`) | *Fine-tuning* completo con lr diferenciado (backbone 1e-4, cabeza 1e-3), AdamW, *warmup* + coseno, *label smoothing* 0,1, dropout 0,2, 15 épocas con *early stopping*. Comparado con *feature extraction* (91,5% vs. 90,7% en valid) |
| Etapa 2, Modelo B | CNN propia (`src/lib/models/cnn_custom.py`) | Stem + 4 bloques convolucionales con BatchNorm y atajos residuales, *global average pooling*, embedding de 512, 5,2 M de parámetros. 60 épocas, AdamW lr 2e-3, dropout 0,3, *weight decay* 5e-4. Tres experimentos (sin augmentation 63,9%, con augmentation 75,3%, con residuales 77,7% en valid) |
| Etapa 3 | YOLOv8n preentrenado en COCO (clase `dog`) | `YOLO_CONF_THRESHOLD=0.25`, elegido comparando 0,10 / 0,25 / 0,40 / 0,50 sobre las escenas del conjunto independiente |

- **Hiperparámetros** en `src/lib/models/train_config.py`; la notebook los sobreescribe por
  experimento con `classifier.train_overrides`. Los checkpoints guardan el modelo completo con la
  lista de razas como atributo (`classes`).
- **Umbral de similitud:** `SIMILARITY_THRESHOLD=0.77` con `cosine`, calibrado con `valid`
  (percentil 0,1 del score del vecino más cercano). Rechaza el 75% de las fotos sin perro y ninguna
  foto de perro que la búsqueda clasificara bien en el conjunto independiente (`2-train.ipynb`, 5.7).
- **Entrenamiento:** local en una GTX 1650 (4 GB), ~1 h en total para todos los experimentos. En
  esa GPU la precisión mixta (fp16) da NaN, así que se entrena en fp32 (`amp: "auto"`); en otras
  GPU (por ejemplo, una T4 de Colab) se activa sola.
- **Checkpoints (solo lectura):** [checkpoints en Google Drive](https://drive.google.com/drive/folders/18upo1D-ivlP3iYNNBukHVn9Lo3OVomuY). Copiar `resnet18_finetuned.pth` y
  `cnn_custom.pth` a `models/`.

## Resultados

Todos los números salen de `2-train.ipynb` (secciones 5 a 14).

**Etapa 1 — búsqueda por similitud** (galería: `train`; Top-10; `cosine` y `l2` dan el mismo ranking
porque los embeddings están normalizados):

| Embeddings | NDCG@10 test | NDCG@10 independiente | Accuracy raza test | Accuracy raza independiente |
|---|---|---|---|---|
| **baseline (ResNet18 ImageNet)** | 0,817 | **0,595** | 0,889 | **0,69** |
| Modelo A (ResNet18 fine-tuned) | **0,888** | 0,572 | **0,907** | 0,64 |
| Modelo B (CNN propia) | 0,730 | 0,248 | 0,760 | 0,29 |

**Etapa 2 — clasificación:**

| Métrica | Modelo A (ResNet18 fine-tuned) | Modelo B (CNN propia) |
|---|---|---|
| Accuracy (test) | **0,906** | 0,762 |
| Precision / Recall / F1 macro (test) | 0,910 / 0,909 / 0,908 | 0,769 / 0,761 / 0,760 |
| Specificity macro (test) | 0,999 | 0,997 |
| Accuracy (conjunto independiente) | 0,57 | 0,27 |
| Parámetros / tamaño | 11,2 M / 42,9 MB | 5,2 M / 19,8 MB |
| Inferencia CPU | ~20 ms/imagen | ~10 ms/imagen |

**Elección:** la app clasifica con el **Modelo A** y busca con los embeddings **`baseline`**: en las
fotos reales del conjunto independiente el baseline rinde igual o mejor que el Modelo A, no depende
de un checkpoint y su umbral está calibrado.

**Comparación entre enfoques:**
- *Fine-tuning vs. CNN propia:* el preentrenamiento en ImageNet es la diferencia. La CNN propia,
  con ~85 imágenes por raza, rinde 14 puntos menos en test y 30 menos en el conjunto
  independiente, aunque es el doble de rápida.
- *Búsqueda vs. clasificación:* la clasificación gana en test (0,906 contra 0,889) y no necesita la
  galería; la búsqueda no requiere entrenar, agrega razas indexando fotos, explica la predicción con
  ejemplos y tiene un rechazo natural (`unknown`). En fotos fuera del dominio del dataset la
  búsqueda con el baseline fue más robusta (0,69 contra 0,57).
- *Detección + clasificación:* clasificar el recorte de YOLO en lugar de la foto entera sube la
  accuracy del Modelo A en el conjunto independiente de 0,57 a 0,63 (de 0,36 a 0,50 cuando el perro
  ocupa menos del 25% de la foto).

**Análisis de errores:** los errores se concentran en razas emparentadas (Lhasa ↔ Shih-Tzu, Cockapoo
↔ Labradoodle, German Sheperd ↔ Malinois, las dos razas sin pelo). Varios errores "seguros" del
Modelo A son etiquetas erróneas del dataset (un "Collie" que es Border Collie, un "Cocker" que es
Cavalier). En el conjunto independiente fallan perros chicos dentro de la escena, en movimiento o en
poses poco comunes. YOLO detecta zorros como perros (falsos positivos) y no detecta perros detrás de
una reja o muy superpuestos.

## Problemas encontrados y soluciones

| Problema | Solución |
|---|---|
| Los splits originales de Kaggle tenían 319 pares de fotos repetidas entre train, valid y test (fuga de datos) | Detección de duplicados por hash perceptual y embedding, y re-particionado sin fuga (`1-pre-processing.ipynb`) |
| El ResNet daba NaN desde el primer paso con precisión mixta en la GTX 1650 | Entrenamiento en fp32 en las GTX 16xx (`amp: "auto"` en `train_config.py`) |
| Leer el dataset desde el disco de Windows en WSL era 3,5 veces más lento | Para entrenar se usó una copia del dataset en el disco de WSL (`TRAIN_DATASET_PATH` en el script de entrenamiento; mismo contenido) |
| WSL no arrancaba por falta de memoria con otras aplicaciones abiertas | Liberar memoria o reiniciar antes de entrenar |
| `CREATE EXTENSION vector` fallaba con *duplicate key* al indexar justo después del primer `docker compose up` (el backend y el script la crean a la vez) | Reintentar `build_index.py`: la extensión ya queda creada |
| El umbral de similitud original (0,55) no rechazaba nada | Calibración con `valid` (0,77) |
| La pestaña de la Etapa 1 permite elegir los embeddings de un modelo entrenado, pero la base guarda los de un solo modelo | Se documenta: para buscar con otro modelo hay que cambiar `EMBEDDING_MODEL` y reindexar |

## Notas importantes

- Para usar `pgvector`, levanten `postgres` y definan `USE_PGVECTOR=true` en `.env`.
- `EMBEDDING_DIM` debe coincidir con la dimensión del embedding del modelo elegido. Si cambian `EMBEDDING_DIM`, la tabla `embeddings` se recrea y se borran los registros previos.
- La base vectorial guarda los embeddings de un modelo a la vez: al cambiar de modelo de embeddings hay que reindexar (`scripts/build_index.py`).
- Las rutas de la galería se guardan relativas a `DATA_PATH` (por ejemplo, `dataset/clean/train/<raza>/<imagen>.jpg`): una base indexada desde el entorno local también sirve para el backend en Docker, porque comparten PostgreSQL y la carpeta `data/`.
- Los modelos (`*.pth`, `*.pt`, `*.onnx`) y los datos no se versionan (ver `.gitignore`). Coloquen los checkpoints en `models/` y súbanlos a un link **público de solo lectura** para que los docentes puedan descargarlos.
- Los pesos preentrenados (torchvision/timm y YOLO) se descargan en el primer uso, por lo que se necesita conexión a internet.

## Entrega

- Pull Request desde su feature branch hacia `main` **de su propio repositorio**, **sin mergear**, con el link subido al campus. No abran el PR contra el repositorio de la cátedra.
- Si el repositorio es privado, los docentes deben tener acceso.
- Notebooks `1-pre-processing.ipynb` y `2-train.ipynb` ejecutadas, con sus salidas.
- Para aprobar, el sistema debe andar sin errores con:

```bash
docker compose build
docker compose up
```

**Fecha de entrega:** a definir por la cátedra. Solo se consideran los commits hasta esa fecha y hora; se recomienda trabajar de forma incremental.

El detalle del contenido del PR, la plantilla de descripción y el checklist están en [ENUNCIADO.md](ENUNCIADO.md#8-entrega-pull-request).
