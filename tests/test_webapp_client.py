from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.webapp import client


def test_top_k_predictions_sorted():
    probs = [0.1, 0.7, 0.2]
    classes = ["bcc", "benign", "melanoma"]
    top = client.top_k_predictions(probs, classes, k=2)
    assert top == [("benign", 0.7), ("melanoma", 0.2)]


def test_call_prediction_api_success(monkeypatch):
    payload = {"label": "bcc", "probs": [0.6, 0.4], "classes": ["bcc", "benign"]}

    class FakeResponse:
        def __init__(self):
            self.status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return payload

    def fake_post(url, files, timeout):
        assert url.endswith("/predict")
        assert "file" in files
        return FakeResponse()

    monkeypatch.setattr(client.requests, "post", fake_post)

    result = client.call_prediction_api(b"img", "test.jpg", api_base_url="http://api")
    assert result.label == "bcc"
    assert result.probs == [0.6, 0.4]


def test_call_prediction_api_error(monkeypatch):
    class BadResponse:
        def raise_for_status(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(client.requests, "post", lambda *args, **kwargs: BadResponse())

    with pytest.raises(client.PredictionError):
        client.call_prediction_api(b"img", "test.jpg", api_base_url="http://api")
