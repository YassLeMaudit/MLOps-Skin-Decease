from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.pipeline_utils import dockerized_python_command

PROJECT_ROOT = Path(os.environ.get("AIRFLOW_PROJECT_ROOT", REPO_ROOT))
PYTHONPATH = str(PROJECT_ROOT)
BASE_ENV = os.environ.copy()
BASE_ENV["PYTHONPATH"] = PYTHONPATH
DEFAULT_ARGS = {
    "owner": "mlops-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="train_model_pipeline",
    description="Sync data from MinIO then retrain the skin disease classifier.",
    default_args=DEFAULT_ARGS,
    schedule_interval="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["training", "mlflow"],
) as dag:
    sync_dataset = BashOperator(
        task_id="sync_dataset",
        bash_command=dockerized_python_command(
            PROJECT_ROOT,
            "src/data/ingest_dataset.py",
            ["--config", "src/data/dataset_config.yaml", "--splits", "train", "val"],
        ),
        env=BASE_ENV,
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=dockerized_python_command(
            PROJECT_ROOT,
            "src/training/train.py",
            [
                "--dataset-config",
                "src/data/dataset_config.yaml",
                "--epochs",
                "{{ dag_run.conf.get('epochs', 5) }}",
                "--batch-size",
                "{{ dag_run.conf.get('batch_size', 16) }}",
                "--learning-rate",
                "{{ dag_run.conf.get('learning_rate', 1e-4) }}",
            ],
        ),
        env=BASE_ENV,
    )

    sync_dataset >> train_model
