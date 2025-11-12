from pathlib import Path

from src.training.pipeline_utils import dockerized_python_command


def test_dockerized_python_command_includes_directory_and_script():
    cmd = dockerized_python_command(Path("/opt/airflow"), "src/data/ingest_dataset.py", ["--foo", "bar"])
    assert cmd == "cd /opt/airflow && python src/data/ingest_dataset.py --foo bar"


def test_dockerized_python_command_without_args():
    cmd = dockerized_python_command("/workspace", "script.py")
    assert cmd == "cd /workspace && python script.py"
