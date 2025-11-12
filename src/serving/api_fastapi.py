import os, io, json, tempfile, zipfile, time
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from dotenv import load_dotenv
from prometheus_client import Counter, Histogram, CONTENT_TYPE_LATEST, generate_latest
load_dotenv()

import tensorflow as tf
import keras
from keras.applications.efficientnet import EfficientNetB0, preprocess_input

# Optionnel: éviter que TF réserve toute la VRAM
for g in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

MODEL_PATH     = os.getenv("MODEL_PATH", "runs/skin5-current/model/model.keras")
CLASS_INDEX    = os.getenv("CLASS_INDEX_PATH", "runs/skin5-current/data/class_index.json")
SIGNATURE_PATH = os.getenv("SIGNATURE_PATH", "runs/skin5-current/inference/signature.json")

app = FastAPI(title="Skin Disease Classifier API (TF/Keras)")
REQUEST_COUNTER = Counter(
    "api_requests_total",
    "Count of HTTP requests handled by the inference API",
    ["method", "endpoint", "http_status"],
)
REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds",
    "Latency of HTTP requests handled by the inference API",
    ["method", "endpoint"],
)
PREDICTION_COUNTER = Counter(
    "api_predictions_total",
    "Number of predictions served by label",
    ["label"],
)
PREDICTION_CONFIDENCE = Histogram(
    "api_prediction_confidence",
    "Confidence score of the top prediction per label",
    ["label"],
    buckets=(0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0),
)
_model = None
_idx_to_name = None
_H = _W = 224


@app.middleware("http")
async def _metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    endpoint = request.url.path
    REQUEST_LATENCY.labels(request.method, endpoint).observe(elapsed)
    REQUEST_COUNTER.labels(request.method, endpoint, str(response.status_code)).inc()
    return response

def _load_meta():
    global _idx_to_name, _H, _W
    with open(CLASS_INDEX, "r", encoding="utf-8") as f:
        class_index = json.load(f)
    _idx_to_name = {i: name for name, i in class_index.items()}
    with open(SIGNATURE_PATH, "r", encoding="utf-8") as f:
        sig = json.load(f)
    _H, _W = int(sig.get("img_height", 224)), int(sig.get("img_width", 224))

def _build_model(num_classes: int) -> keras.Model:
    base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(_H, _W, 3))
    x = keras.layers.GlobalAveragePooling2D()(base.output)
    x = keras.layers.Dropout(0.30)(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.30)(x)
    out = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs=base.input, outputs=out)

def _try_load_keras3():
    # tentative 1 : chargement direct (si ça marche chez toi, on garde)
    return keras.saving.load_model(MODEL_PATH)

def _fallback_from_weights():
    # tentative 2 : ouvrir le .keras comme zip et charger uniquement les poids
    if not zipfile.is_zipfile(MODEL_PATH):
        raise RuntimeError("Le fichier MODEL_PATH n'est pas un .keras (zip).")
    with zipfile.ZipFile(MODEL_PATH, "r") as z:
        if "model.weights.h5" not in z.namelist():
            raise RuntimeError("model.weights.h5 absent dans l'archive .keras")
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp.write(z.read("model.weights.h5"))
            weights_path = tmp.name

    num_classes = len(_idx_to_name)
    model = _build_model(num_classes)
    # Chargement tolérant : par nom + skip des couches non correspondantes
    model.load_weights(weights_path, by_name=True, skip_mismatch=True)
    try:
        os.remove(weights_path)
    except Exception:
        pass
    return model

def _ensure_loaded():
    global _model
    if _model is not None:
        return
    _load_meta()
    # on tente d'abord le chargement natif Keras 3
    try:
        _model = _try_load_keras3()
    except Exception as e:
        print("[warn] keras.saving.load_model a échoué -> fallback poids uniquement:", e)
        _model = _fallback_from_weights()

def _preprocess(pil_img: Image.Image):
    img = pil_img.convert("RGB").resize((_W, _H))
    arr = np.array(img).astype("float32")
    arr = preprocess_input(arr)
    return arr[np.newaxis, ...]

@app.get("/health")
def health():
    ok = all(os.path.exists(p) for p in [MODEL_PATH, CLASS_INDEX, SIGNATURE_PATH])
    return {
        "status": "ok" if ok else "missing_artifacts",
        "model_path": MODEL_PATH,
        "gpu_devices": [d.name for d in tf.config.list_physical_devices("GPU")],
    }


@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        _ensure_loaded()
        img = Image.open(io.BytesIO(await file.read()))
        x = _preprocess(img)
        probs = _model.predict(x, verbose=0).squeeze().tolist()
        pred = int(np.argmax(probs))
        classes = [_idx_to_name[i] for i in range(len(_idx_to_name))]
        label = _idx_to_name.get(pred, str(pred))
        best_prob = float(probs[pred])
        PREDICTION_COUNTER.labels(label=label).inc()
        PREDICTION_CONFIDENCE.labels(label=label).observe(best_prob)
        return {"label": label, "probs": probs, "classes": classes}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
