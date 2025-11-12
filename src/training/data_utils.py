from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple
from urllib.parse import urlparse

from src.data.ingest_dataset import METADATA_TABLE

Record = Dict[str, str]


def fetch_records(db_path: Path, dataset_name: str, splits: Iterable[str]) -> List[Record]:
    """Return metadata rows for the provided dataset and splits."""
    split_list = list(splits)
    if not split_list:
        raise ValueError("At least one split must be provided.")
    placeholders = ",".join(["?"] * len(split_list))
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(
        f"""
        SELECT split, label, source_path, s3_uri, sha256
        FROM {METADATA_TABLE}
        WHERE dataset_name=? AND split IN ({placeholders})
        ORDER BY id ASC
        """,
        [dataset_name, *split_list],
    )
    rows = [
        {
            "split": row[0],
            "label": row[1],
            "source_path": row[2],
            "s3_uri": row[3],
            "sha256": row[4],
        }
        for row in cursor.fetchall()
    ]
    conn.close()
    return rows


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Unsupported URI scheme for {uri}")
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    return bucket, key


def ensure_local_file(
    record: Record,
    cache_dir: Path,
    download_fn: Callable[[str, str, Path], None],
) -> Path:
    """Return a path to the image, downloading from S3 if needed."""
    source_path = record.get("source_path")
    if source_path:
        candidate = Path(source_path)
        if candidate.exists():
            return candidate
    s3_uri = record.get("s3_uri")
    if not s3_uri:
        raise FileNotFoundError(f"No local path or s3_uri for record {record}")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(urlparse(s3_uri).path).suffix or ".img"
    sha = record.get("sha256") or "nohash"
    destination = cache_dir / f"{sha}{suffix}"
    if destination.exists():
        return destination
    bucket, key = parse_s3_uri(s3_uri)
    download_fn(bucket, key, destination)
    return destination
