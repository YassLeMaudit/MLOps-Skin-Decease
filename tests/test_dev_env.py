from __future__ import annotations

import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_expected_directories_exist():
    required = [
        "src/serving",
        "src/webapp",
        "src/training",
        "src/data",
        "docker",
        "airflow/dags",
        "airflow/logs",
        "airflow/plugins",
        "mlruns",
    ]
    missing = [str(REPO_ROOT / path) for path in required if not (REPO_ROOT / path).exists()]
    assert not missing, f"Missing expected directories: {missing}"


def test_env_file_declares_core_services():
    env_path = REPO_ROOT / ".env"
    assert env_path.exists(), ".env file is required for local dev"
    env_vars: dict[str, str] = {}
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        env_vars[key.strip()] = value.strip()

    expected_keys = {
        "MLFLOW_TRACKING_URI",
        "MLFLOW_S3_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "MINIO_ROOT_USER",
        "MINIO_ROOT_PASSWORD",
        "AIRFLOW_IMAGE",
        "AIRFLOW__CORE__EXECUTOR",
        "AIRFLOW__CORE__FERNET_KEY",
        "AIRFLOW__WEBSERVER__SECRET_KEY",
        "API_PORT",
        "WEBAPP_PORT",
        "MLFLOW_PORT",
        "MINIO_API_PORT",
        "MINIO_CONSOLE_PORT",
    }
    missing = sorted(key for key in expected_keys if key not in env_vars)
    assert not missing, f".env is missing keys: {missing}"


def test_compose_declares_all_required_services():
    compose_path = REPO_ROOT / "compose.yaml"
    assert compose_path.exists(), "compose.yaml must exist"
    compose = yaml.safe_load(compose_path.read_text())
    services = compose.get("services", {})
    required_services = {
        "mlflow",
        "mlflow-db",
        "minio",
        "minio-mc",
        "api",
        "webapp",
        "airflow-db",
        "airflow-webserver",
        "airflow-scheduler",
        "airflow-worker",
        "redis",
        "airflow-init",
    }
    missing = sorted(name for name in required_services if name not in services)
    assert not missing, f"compose.yaml missing services: {missing}"

    api_service = services["api"]
    build = api_service.get("build") or {}
    assert build.get("dockerfile") == "docker/Dockerfile.api", "API service must use Dockerfile.api"
    assert api_service.get("ports"), "API service should expose a port for dev access"
