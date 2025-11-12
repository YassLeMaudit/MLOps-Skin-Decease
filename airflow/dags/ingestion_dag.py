from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_DIR = REPO_ROOT / "src"

from src.data.ingest_dataset import run_ingestion


def _ingest():
    config_path = SRC_DIR / "data/dataset_config.yaml"
    run_ingestion(config_path)


with DAG(
    dag_id="ingest_dataset_to_minio",
    description="Sync local Kaggle dataset into MinIO buckets",
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["ingestion", "minio", "dataset"],
) as dag:
    ingest_task = PythonOperator(
        task_id="ingest_local_dataset",
        python_callable=_ingest,
    )
