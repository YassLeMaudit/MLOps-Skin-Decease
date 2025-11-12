from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from src.data.ingest_dataset import (
    DatasetConfig,
    FileRecord,
    SplitConfig,
    discover_files,
    ingest_records,
    load_dataset_config,
)


def write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(
        """
dataset:
  name: demo
  labels: [bcc, benign]
  splits:
    train:
      root: TRAIN_ROOT
s3:
  bucket_env: TEST_BUCKET_ENV
  prefix: demo
metadata:
  sqlite_path: METADATA_DB
        """.replace(
            "TRAIN_ROOT", str(tmp_path / "train")
        ).replace(
            "METADATA_DB", str(tmp_path / "db.sqlite")
        ),
        encoding="utf-8",
    )
    return config_path


def test_load_dataset_config_uses_env_bucket(tmp_path, monkeypatch):
    config_path = write_config(tmp_path)
    monkeypatch.setenv("TEST_BUCKET_ENV", "unit-bucket")
    cfg = load_dataset_config(config_path)
    assert cfg.s3_bucket == "unit-bucket"
    assert "train" in cfg.splits
    assert cfg.metadata_db.name == "db.sqlite"


def test_discover_files_picks_supported_extensions(tmp_path, monkeypatch):
    config_path = write_config(tmp_path)
    monkeypatch.setenv("TEST_BUCKET_ENV", "unit-bucket")
    cfg = load_dataset_config(config_path)
    train_root = cfg.splits["train"].root
    (train_root / "bcc").mkdir(parents=True)
    (train_root / "benign").mkdir()
    (train_root / "bcc" / "img1.jpg").write_bytes(b"1")
    (train_root / "bcc" / "note.txt").write_text("ignore", encoding="utf-8")
    (train_root / "benign" / "img2.png").write_bytes(b"2")

    records = discover_files(cfg, splits=["train"])
    assert len(records) == 2
    splits = {rec.split for rec in records}
    labels = {rec.label for rec in records}
    assert splits == {"train"}
    assert labels == {"bcc", "benign"}


def test_ingest_records_uploads_and_persists(tmp_path, monkeypatch):
    train_root = tmp_path / "train"
    label_dir = train_root / "bcc"
    label_dir.mkdir(parents=True)
    img_path = label_dir / "img.jpg"
    img_path.write_bytes(b"fake-image")

    cfg = DatasetConfig(
        dataset_name="demo",
        description="",
        labels=["bcc"],
        splits={"train": SplitConfig(name="train", root=train_root)},
        s3_bucket="bucket-demo",
        s3_prefix="demo",
        metadata_db=tmp_path / "db.sqlite",
    )

    uploaded = {}

    class FakeS3:
        def upload_file(self, filename, bucket, key):
            uploaded["filename"] = filename
            uploaded["bucket"] = bucket
            uploaded["key"] = key

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("MINIO_REGION", "us-east-1")
    monkeypatch.setattr("src.data.ingest_dataset.create_s3_client", lambda: FakeS3())

    record = FileRecord(split="train", label="bcc", path=img_path)
    stats = ingest_records(cfg, [record], dry_run=False)
    assert stats["train:bcc"] == 1
    assert uploaded["bucket"] == "bucket-demo"
    assert uploaded["filename"] == str(img_path)

    conn = sqlite3.connect(cfg.metadata_db)
    row = conn.execute("SELECT split, label FROM lesions_data").fetchone()
    conn.close()
    assert row == ("train", "bcc")
