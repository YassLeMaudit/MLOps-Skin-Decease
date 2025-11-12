from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import requests


class PredictionError(RuntimeError):
    """Raised when the prediction API fails."""


@dataclass
class PredictionResult:
    label: str
    probs: List[float]
    classes: List[str]

    @property
    def paired(self) -> List[Tuple[str, float]]:
        return list(zip(self.classes, self.probs))


def call_prediction_api(
    image_bytes: bytes,
    filename: str,
    api_base_url: str | None = None,
    timeout: int = 30,
) -> PredictionResult:
    base_url = api_base_url or os.getenv("API_BASE_URL", "http://localhost:8000")
    url = f"{base_url.rstrip('/')}/predict"
    files = {"file": (filename, io.BytesIO(image_bytes), "application/octet-stream")}
    try:
        response = requests.post(url, files=files, timeout=timeout)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        raise PredictionError(f"API request failed: {exc}") from exc

    try:
        payload = response.json()
    except Exception as exc:  # noqa: BLE001
        raise PredictionError("Could not parse API response.") from exc
    if not {"label", "probs", "classes"} <= payload.keys():
        raise PredictionError("Malformed response from API.")
    return PredictionResult(
        label=payload["label"],
        probs=list(payload["probs"]),
        classes=list(payload["classes"]),
    )


def top_k_predictions(
    probs: Sequence[float],
    classes: Sequence[str],
    k: int,
) -> List[Tuple[str, float]]:
    pairs = list(zip(classes, probs))
    pairs.sort(key=lambda item: item[1], reverse=True)
    return pairs[: max(k, 0)]
