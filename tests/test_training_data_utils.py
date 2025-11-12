from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src.data.ingest_dataset import METADATA_TABLE
from src.training.data_utils import ensure_local_file, fetch_records, parse_s3_uri


def setup_db(tmp_path: Path):
    db_path = tmp_path / "db.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute(
        f"""
        CREATE TABLE {METADATA_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_name TEXT,
            split TEXT,
            label TEXT,
            source_path TEXT,
            s3_uri TEXT,
            sha256 TEXT,
            bytes INTEGER
        )
        """
    )
    conn.commit()
    return conn, db_path


def test_fetch_records_returns_rows(tmp_path):
    conn, db_path = setup_db(tmp_path)
    conn.execute(
        f"INSERT INTO {METADATA_TABLE}(dataset_name, split, label, source_path, s3_uri, sha256, bytes) VALUES (?,?,?,?,?,?,?)",
        ("demo", "train", "bcc", "/tmp/img1.jpg", "s3://bucket/a.jpg", "sha1", 10),
    )
    conn.execute(
        f"INSERT INTO {METADATA_TABLE}(dataset_name, split, label, source_path, s3_uri, sha256, bytes) VALUES (?,?,?,?,?,?,?)",
        ("demo", "val", "benign", "/tmp/img2.jpg", "s3://bucket/b.jpg", "sha2", 12),
    )
    conn.commit()
    conn.close()

    train_records = fetch_records(db_path, "demo", ["train"])
    assert len(train_records) == 1
    assert train_records[0]["label"] == "bcc"


def test_ensure_local_file_prefers_existing_path(tmp_path):
    img = tmp_path / "img.jpg"
    img.write_bytes(b"binary")
    record = {
        "source_path": str(img),
        "s3_uri": "s3://bucket/a.jpg",
        "sha256": "sha",
    }

    called = False

    def fake_download(bucket, key, destination):
        nonlocal called
        called = True

    result = ensure_local_file(record, tmp_path / "cache", fake_download)
    assert result == img
    assert called is False


def test_ensure_local_file_downloads_when_missing(tmp_path):
    cache_dir = tmp_path / "cache"
    record = {
        "source_path": str(tmp_path / "missing.jpg"),
        "s3_uri": "s3://bucket/train/a.jpg",
        "sha256": "abc",
    }

    def fake_download(bucket, key, destination):
        assert bucket == "bucket"
        assert key == "train/a.jpg"
        destination.write_bytes(b"downloaded")

    result = ensure_local_file(record, cache_dir, fake_download)
    assert result.exists()
    assert result.read_bytes() == b"downloaded"


def test_parse_s3_uri():
    bucket, key = parse_s3_uri("s3://demo/path/to/file.jpg")
    assert bucket == "demo"
    assert key == "path/to/file.jpg"
