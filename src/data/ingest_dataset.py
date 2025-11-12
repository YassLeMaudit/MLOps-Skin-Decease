from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import boto3
import yaml

LOGGER = logging.getLogger("ingestion")
DEFAULT_EXTENSIONS = {".jpg", ".jpeg", ".png"}
METADATA_TABLE = "lesions_data"


@dataclass(frozen=True)
class SplitConfig:
    name: str
    root: Path


@dataclass(frozen=True)
class DatasetConfig:
    dataset_name: str
    description: str
    labels: List[str]
    splits: Dict[str, SplitConfig]
    s3_bucket: str
    s3_prefix: str
    metadata_db: Path


@dataclass
class FileRecord:
    split: str
    label: str
    path: Path
    sha256: Optional[str] = None
    size: Optional[int] = None


def load_dataset_config(path: Path) -> DatasetConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    dataset = data.get("dataset") or {}
    s3_section = data.get("s3") or {}
    metadata_section = data.get("metadata") or {}

    bucket = s3_section.get("bucket")
    bucket_env = s3_section.get("bucket_env")
    if bucket_env:
        bucket = os.getenv(bucket_env, bucket)
    if not bucket:
        raise ValueError("S3 bucket is not defined in config or environment.")

    splits_data = dataset.get("splits") or {}
    splits: Dict[str, SplitConfig] = {}
    for name, spec in splits_data.items():
        root = Path(spec["root"]).expanduser()
        splits[name] = SplitConfig(name=name, root=root)

    labels = dataset.get("labels") or []
    metadata_db = Path(metadata_section.get("sqlite_path", "data/metadata/skin_metadata.db")).expanduser()
    metadata_db.parent.mkdir(parents=True, exist_ok=True)

    return DatasetConfig(
        dataset_name=dataset.get("name", "dataset"),
        description=dataset.get("description", ""),
        labels=labels,
        splits=splits,
        s3_bucket=bucket,
        s3_prefix=s3_section.get("prefix", dataset.get("name", "dataset")).rstrip("/"),
        metadata_db=metadata_db,
    )


def create_s3_client():
    endpoint = (
        os.getenv("S3_ENDPOINT_URL")
        or os.getenv("MLFLOW_S3_ENDPOINT_URL")
        or "http://localhost:9000"
    )
    session = boto3.session.Session(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("MINIO_REGION", "us-east-1"),
    )
    return session.client("s3", endpoint_url=endpoint)


def discover_files(config: DatasetConfig, splits: Optional[Iterable[str]] = None) -> List[FileRecord]:
    selected_splits = list(splits) if splits else list(config.splits.keys())
    records: List[FileRecord] = []
    for split_name in selected_splits:
        split_cfg = config.splits.get(split_name)
        if not split_cfg:
            raise ValueError(f"Unknown split '{split_name}' in config.")
        for label in config.labels:
            label_dir = split_cfg.root / label
            if not label_dir.exists():
                LOGGER.warning("Label directory missing: %s", label_dir)
                continue
            for file_path in label_dir.rglob("*"):
                if not file_path.is_file():
                    continue
                if file_path.suffix.lower() not in DEFAULT_EXTENSIONS:
                    continue
                records.append(FileRecord(split=split_name, label=label, path=file_path))
    return records


def ensure_metadata_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {METADATA_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_name TEXT NOT NULL,
            split TEXT NOT NULL,
            label TEXT NOT NULL,
            source_path TEXT NOT NULL,
            s3_uri TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            bytes INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(dataset_name, split, sha256)
        )
        """
    )
    conn.commit()


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_s3_key(config: DatasetConfig, record: FileRecord) -> str:
    suffix = record.path.suffix.lower()
    stem = record.path.stem[:32]
    sha = record.sha256[:10] if record.sha256 else "unknown"
    return f"{config.s3_prefix}/{record.split}/{record.label}/{stem}-{sha}{suffix}"


def record_exists(conn: sqlite3.Connection, cfg: DatasetConfig, record: FileRecord) -> bool:
    cur = conn.execute(
        f"SELECT 1 FROM {METADATA_TABLE} WHERE dataset_name=? AND split=? AND sha256=? LIMIT 1",
        (cfg.dataset_name, record.split, record.sha256),
    )
    return cur.fetchone() is not None


def ingest_records(
    config: DatasetConfig,
    records: List[FileRecord],
    dry_run: bool = False,
) -> Dict[str, int]:
    conn = sqlite3.connect(config.metadata_db)
    ensure_metadata_table(conn)
    s3 = None if dry_run else create_s3_client()
    stats: Dict[str, int] = {}

    for record in records:
        record.sha256 = record.sha256 or compute_sha256(record.path)
        record.size = record.size or record.path.stat().st_size

        if record_exists(conn, config, record):
            LOGGER.info(
                "Skipping existing file (split=%s label=%s sha=%s)",
                record.split,
                record.label,
                record.sha256,
            )
            continue

        key = build_s3_key(config, record)
        s3_uri = f"s3://{config.s3_bucket}/{key}"

        if not dry_run:
            s3.upload_file(str(record.path), config.s3_bucket, key)
        conn.execute(
            f"""
            INSERT INTO {METADATA_TABLE}(dataset_name, split, label, source_path, s3_uri, sha256, bytes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                config.dataset_name,
                record.split,
                record.label,
                str(record.path),
                s3_uri,
                record.sha256,
                record.size,
            ),
        )
        stats_key = f"{record.split}:{record.label}"
        stats[stats_key] = stats.get(stats_key, 0) + 1
        LOGGER.info("Uploaded %s -> %s", record.path, s3_uri)

    conn.commit()
    conn.close()
    return stats


def run_ingestion(config_path: Path, splits: Optional[List[str]] = None, dry_run: bool = False) -> Dict[str, int]:
    cfg = load_dataset_config(config_path)
    records = discover_files(cfg, splits)
    return ingest_records(cfg, records, dry_run=dry_run)


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Ingest local Kaggle dataset into MinIO/S3.")
    parser.add_argument("--config", type=Path, default=Path("src/data/dataset_config.yaml"))
    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="Optional list of splits (train, val, ...) to ingest. Defaults to all splits.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Compute metadata without touching S3.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    stats = run_ingestion(args.config, args.splits, dry_run=args.dry_run)
    LOGGER.info("Ingestion summary:")
    for key, count in sorted(stats.items()):
        split, label = key.split(":")
        LOGGER.info("  %s / %s -> %d files", split, label, count)


if __name__ == "__main__":
    main()
