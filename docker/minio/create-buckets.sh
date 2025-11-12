#!/usr/bin/env bash
set -euo pipefail

echo "[minio-init] Waiting for MinIO to be reachable..."
until mc alias set local http://minio:9000 "${MINIO_ROOT_USER}" "${MINIO_ROOT_PASSWORD}"; do
  sleep 2
done

IFS=',' read -ra buckets <<< "${MINIO_DEFAULT_BUCKETS:-skin}"
for bucket in "${buckets[@]}"; do
  name="${bucket#"${bucket%%[![:space:]]*}"}"
  name="${name%"${name##*[![:space:]]}"}"
  if [[ -z "$name" ]]; then
    continue
  fi
  echo "[minio-init] Ensuring bucket '$name'"
  mc mb -p "local/${name}" >/dev/null 2>&1 || true
  mc anonymous set download "local/${name}" >/dev/null 2>&1 || true
done

echo "[minio-init] Buckets ready."
