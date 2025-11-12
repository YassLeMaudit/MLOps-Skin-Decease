# MLOps Skin Disease Platform

This repository hosts the end-to-end workflow for the skin disease classification project: data ingestion, training with MLflow tracking, FastAPI serving, Streamlit webapp, and Airflow orchestration.

## Step 1 - Local Dev Environment

The `.env` file (already versioned for the dev stack) centralizes credentials, service ports, and bucket names. Feel free to edit values locally if ports clash.

### Prerequisites

- Docker Engine & Docker Compose v2+
- Python 3.12 with `uv` (already used in the repo) for local scripts/tests

### Bring up the stack

```bash
docker compose up --build
```

Exposed endpoints once the stack is healthy:

| Service  | URL                         |
|----------|-----------------------------|
| FastAPI  | http://localhost:8000/docs  |
| Webapp   | http://localhost:8501       |
| MLflow   | http://localhost:5000       |
| MinIO    | http://localhost:9001       |
| Airflow  | http://localhost:8080       |
| Prometheus | http://localhost:9090     |
| Grafana  | http://localhost:3000       |

**Default credentials**

| Component | User / Password | Notes |
|-----------|-----------------|-------|
| Airflow   | `admin` / `admin` | Can be reset via `airflow users create` in `airflow-init`. |
| Grafana   | `admin` / `admin` | Change after first login via UI if deploying publicly. |
| MinIO     | `${MINIO_ROOT_USER}` / `${MINIO_ROOT_PASSWORD}` (see `.env`) | Also reused for MLflow artifact access. |

**Start commands**

```bash
# Core platform (API, webapp, MLflow, MinIO, Airflow, monitoring)
docker compose up -d \
  minio minio-mc mlflow-db mlflow \
  api webapp \
  airflow-db redis airflow-init airflow-webserver airflow-scheduler airflow-worker \
  prometheus grafana

# Follow logs for Airflow or API during debugging
docker compose logs -f airflow-webserver
docker compose logs -f api
```

The compose file wires volumes so that code changes on the host are reflected inside the API/webapp containers (hot reload friendly). Airflow shares the local `airflow/` directory for DAG authoring.

### Tests

Run smoke checks for the environment scaffolding:

```bash
uv run pytest tests/test_dev_env.py
```

These tests make sure the `.env` file exposes the required keys and the compose stack defines every mandatory service before we move to the next stages.

## Step 2 - Data Ingestion

### Dataset layout

Place the Kaggle dataset under `data/raw/` as already done:

```
data/raw/
|-- train/<label>/*.jpg
`-- val/<label>/*.jpg
```

The configuration file `src/data/dataset_config.yaml` points to these folders, lists the five labels, and defines where metadata is stored (`data/metadata/skin_metadata.db`) plus the MinIO bucket/prefix used for uploads.

### Local ingestion script

```
uv run python -m src.data.ingest_dataset \
  --config src/data/dataset_config.yaml \
  --splits train val
```

What it does:

- Scans each split/label folder, ignoring non-image files.
- Computes SHA-256 hashes for deduplication and stores metadata in `lesions_data` (SQLite).
- Uploads every file to MinIO (`s3://skin-processed/skin/<split>/<label>/...`) using the credentials declared in `.env`.
- Prints a per-label summary.

Set `--dry-run` to only refresh metadata without touching MinIO.

### Airflow DAG

The DAG `airflow/dags/ingestion_dag.py` imports `run_ingestion` and runs daily (`@daily`). It expects the shared volume `/opt/airflow/src` (already wired by Compose) so it can read the same config file. Trigger it from the Airflow UI to re-sync MinIO or integrate it into a broader pipeline later.

### Tests

```
uv run python -m pytest tests/test_dev_env.py tests/test_ingestion.py
```

`tests/test_ingestion.py` validates config parsing, file discovery, and that ingesting a file records metadata + calls the mocked S3 client.

## Step 3 - Training + MLflow

### Script

```
uv run python -m src.training.train \
  --dataset-config src/data/dataset_config.yaml \
  --epochs 5 \
  --batch-size 16 \
  --learning-rate 1e-4
```

What happens:

- Lit la base `data/metadata/skin_metadata.db`, récupère les chemins des splits `train` / `val` et reconstruit le mapping `label -> index`.
- Vérifie l’existence locale des images ou les télécharge depuis MinIO dans `.cache/dataset`.
- Construit un pipeline `tf.data` + EfficientNetB0 (en gelant le backbone), entraîne sur GPU si présent.
- Enregistre paramètres/métriques dans MLflow (accuracy, F1 macro), log le modèle Keras comme artefact et exporte `runs/skin5-current/`.
- Charge le modèle final dans `S3_MODELS_BUCKET` (`models-registry` par défaut) sous `skin/models/<run_id>/model.keras`.

### Résultats

`runs/skin5-current/` contient :

- `data/class_index.json`
- `inference/signature.json`
- `model/model.keras`

Les mêmes fichiers sont loggés dans MLflow pour audit. Le dossier `.cache/dataset` peut être vidé quand tu veux.

### Tests

```
uv run python -m pytest \
  tests/test_dev_env.py \
  tests/test_ingestion.py \
  tests/test_training_data_utils.py
```

Le dernier jeu couvre les utilitaires SQLite/S3 utilisés par `train.py`.

## Step 4 - Streamlit Webapp

### Run locally

```
uv run streamlit run src/webapp/app_streamlit.py --server.port=8501 --server.address=0.0.0.0
```

Set `API_BASE_URL` in `.env` (default `http://localhost:8000`). Inside Docker Compose, the variable is overridden to `http://api:8000`, so the app can talk to the FastAPI container.

### Features

- Upload JPEG/PNG, call `POST /predict`, and display the predicted class plus a configurable Top-K list.
- Adjustable uncertainty threshold (default 0.5). If the best probability is below the threshold, the UI highlights a warning inviting the user to seek medical confirmation.
- Uses `src/webapp/client.py` to isolate HTTP logic and keep Streamlit code lean.

### Tests

```
uv run python -m pytest \
  tests/test_dev_env.py \
  tests/test_ingestion.py \
  tests/test_training_data_utils.py \
  tests/test_webapp_client.py
```

`tests/test_webapp_client.py` validates the Top-K helper and the HTTP client error handling (mocked requests).

## Step 6 - Airflow Training DAG

- DAG `train_model_pipeline` dans `airflow/dags/train_pipeline_dag.py`.
- Pipeline : `sync_dataset` (ré-ingestion depuis MinIO) puis `train_model` (exécute `src/training/train.py`).
- Déploiement : `docker compose up -d airflow-webserver airflow-scheduler airflow-worker`.
- UI : http://localhost:8080 (admin/admin). Active le DAG, puis clique sur “Trigger DAG” pour lancer un run manuel ou laisse le scheduler `@weekly`.
- Paramètres optionnels via `Trigger DAG > JSON conf`, ex. `{"epochs":10,"batch_size":32,"learning_rate":0.0005}` pour ajuster l’entraînement.
- En ligne de commande (à lancer dans le conteneur `airflow-webserver` ou sur ta machine si `airflow` est installé) :

```bash
airflow dags trigger train_model_pipeline \
  --conf '{"epochs": 10, "batch_size": 32, "learning_rate": 5e-4}'
```

Les journaux pour chaque tâche se consultent via l’onglet *Graph* → clic sur la tâche → *Logs*.

## Step 7 - Monitoring & Observability

The monitoring stack lives under `monitoring/` and is wired directly in `docker compose`:

| Service    | URL                      | Description |
|------------|--------------------------|-------------|
| Prometheus | http://localhost:9090    | Scrapes Airflow `/admin/metrics`, the FastAPI `/metrics`, the Streamlit UI exporter, and itself |
| Grafana    | http://localhost:3000    | Default credentials `admin/admin`, auto-provisioned Prometheus datasource + “Airflow Overview” dashboard |

### FastAPI metrics
- `/metrics` endpoint on the API container powered by `prometheus-client`.
- `api_requests_total{endpoint,method,status}` and `api_request_duration_seconds` track usage & latency.
- `api_predictions_total{label}` and `api_prediction_confidence{label}` expose served diagnoses and confidence.
- `fastapi_request_inprogress` (native Starlette collector) counts concurrent requests.

### Streamlit metrics
- A tiny Prometheus exporter (port `9100` inside the container) started by the webapp.
- Counters for page views/uploads/button clicks, histogram for round-trip latency to the API, and per-label display counts.
- Metric names: `webapp_page_views_total`, `webapp_uploads_total`, `webapp_predict_clicks_total`, `webapp_prediction_results_total{label}`, `webapp_prediction_latency_seconds{status}`.

### Airflow & infrastructure
- Airflow webserver exposes `/admin/metrics`: uptime gauge, queue depth, scheduler loop timing, DAG run/task metrics.
- Prometheus job `prometheus` self-scrapes to monitor scrape health.

Start/stop the monitoring stack alongside the app:

```bash
docker compose up -d prometheus grafana
```

## Step 8 - CI/CD

GitHub Actions workflow `.github/workflows/ci.yml` automates regression tests and Docker builds:

1. **Lint & tests job**
   - Runs on every push/PR targeting `main`.
   - Sets up Python 3.12, caches pip, installs the project (`pip install -e .`) then executes `pytest`.
2. **Docker build job**
   - Builds `docker/Dockerfile.api` and `docker/Dockerfile.app` after tests succeed.
   - If `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN` secrets are defined and the push targets `main`, images are tagged with `${GITHUB_SHA}` and pushed to your Docker Hub namespace.

Manual trigger (`workflow_dispatch`) lets you rebuild images on demand. Adapt the workflow (extra jobs, staging deploy, etc.) by editing `.github/workflows/ci.yml`.

To push via CI:

1. Définis les secrets GitHub `DOCKERHUB_USERNAME` et `DOCKERHUB_TOKEN`.
2. Merge/push sur `main`.
3. Le job `docker-build` taggera automatiquement `skin-mlops-api:${GITHUB_SHA}` et `skin-mlops-webapp:${GITHUB_SHA}` puis les publiera sur ton namespace Docker Hub.
