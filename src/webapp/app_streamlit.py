from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from prometheus_client import Counter, Histogram, REGISTRY, start_http_server

# Allow running `streamlit run` without installing the package
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.webapp.client import PredictionError, call_prediction_api, top_k_predictions

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
WEBAPP_METRICS_PORT = int(os.getenv("WEBAPP_METRICS_PORT", "9100"))

if not os.environ.get("WEBAPP_METRICS_SERVER_STARTED"):
    start_http_server(WEBAPP_METRICS_PORT, addr="0.0.0.0")
    os.environ["WEBAPP_METRICS_SERVER_STARTED"] = "1"


def _get_or_create_metric(name: str, factory):
    existing = getattr(REGISTRY, "_names_to_collectors", {}).get(name)
    if existing:
        return existing
    metric = factory()
    return metric


PAGE_VIEWS = _get_or_create_metric(
    "webapp_page_views_total",
    lambda: Counter("webapp_page_views_total", "Total times the Streamlit page was rendered"),
)
UPLOAD_COUNTER = _get_or_create_metric(
    "webapp_uploads_total",
    lambda: Counter("webapp_uploads_total", "Number of uploaded images handled by the UI"),
)
PREDICT_CLICKS = _get_or_create_metric(
    "webapp_predict_clicks_total",
    lambda: Counter("webapp_predict_clicks_total", "Number of times the Predict button was pressed"),
)
PREDICTION_RESULTS = _get_or_create_metric(
    "webapp_prediction_results_total",
    lambda: Counter(
        "webapp_prediction_results_total",
        "Predicted labels displayed to users",
        ["label"],
    ),
)
API_LATENCY = _get_or_create_metric(
    "webapp_prediction_latency_seconds",
    lambda: Histogram(
        "webapp_prediction_latency_seconds",
        "Latency of calls from the UI to the FastAPI backend",
        ["status"],
        buckets=(0.5, 1, 2, 4, 8, 16),
    ),
)

st.set_page_config(page_title="Skin Disease Classifier", page_icon="🩺", layout="centered")
PAGE_VIEWS.inc()

st.title("Skin Disease Detection")
st.caption("Upload a skin lesion image to get model predictions via the FastAPI backend.")

col_left, col_right = st.columns(2)
with col_left:
    threshold = st.slider("Uncertainty threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
with col_right:
    top_k = st.number_input("Top-K predictions", min_value=1, max_value=5, value=3, step=1)

uploaded = st.file_uploader("Select an image", type=["jpg", "jpeg", "png"])

if uploaded:
    st.image(uploaded, caption="Uploaded image", use_column_width=True)
    if st.button("Predict", type="primary"):
        UPLOAD_COUNTER.inc()
        PREDICT_CLICKS.inc()
        with st.spinner("Contacting API..."):
            try:
                start = time.perf_counter()
                result = call_prediction_api(uploaded.getvalue(), uploaded.name, api_base_url=API_BASE_URL)
            except PredictionError as exc:
                API_LATENCY.labels(status="error").observe(time.perf_counter() - start)
                st.error(str(exc))
            else:
                duration = time.perf_counter() - start
                API_LATENCY.labels(status="success").observe(duration)
                pairs = top_k_predictions(result.probs, result.classes, k=top_k)
                best_label, best_prob = pairs[0]
                PREDICTION_RESULTS.labels(label=best_label).inc()
                st.subheader(f"Predicted: **{best_label}** ({best_prob:.2%})")

                if best_prob < threshold:
                    st.warning(
                        "Model uncertainty detected — consider sending this case to a dermatologist for confirmation."
                    )

                st.markdown("### Top predictions")
                for label, prob in pairs:
                    st.write(f"- **{label}**: {prob:.2%}")
else:
    st.info("Waiting for an image upload to start the prediction.")
