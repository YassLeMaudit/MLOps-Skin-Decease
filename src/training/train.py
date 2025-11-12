from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Optional

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_ENABLE_XLA", "0")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from tensorflow import keras

tf.config.optimizer.set_jit(False)
for gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


class MLflowLoggingCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for key, value in logs.items():
            if value is None:
                continue
            mlflow.log_metric(key, float(value), step=epoch)

from src.data.ingest_dataset import create_s3_client, load_dataset_config
from src.training.data_utils import ensure_local_file, fetch_records

AUTO = tf.data.AUTOTUNE


def build_model(img_height: int, img_width: int, num_classes: int, dropout: float = 0.3) -> keras.Model:
    inputs = keras.Input(shape=(img_height, img_width, 3))
    x = keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def build_dataset(
    paths: List[str],
    labels: List[int],
    img_height: int,
    img_width: int,
    batch_size: int,
    shuffle: bool,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), reshuffle_each_iteration=True)

    def _load(path, label):
        image = tf.io.read_file(path)
        image = tf.io.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, (img_height, img_width))
        image = tf.cast(image, tf.float32)
        image = keras.applications.efficientnet.preprocess_input(image)
        return image, label

    ds = ds.map(_load, num_parallel_calls=AUTO)
    ds = ds.batch(batch_size)
    return ds.prefetch(AUTO)


def make_download_fn(s3_client):
    def _download(bucket: str, key: str, destination: Path):
        s3_client.download_file(bucket, key, str(destination))

    return _download


def export_artifacts(
    export_dir: Path,
    labels: List[str],
    img_height: int,
    img_width: int,
    model: keras.Model,
) -> Path:
    export_dir = export_dir.resolve()
    (export_dir / "data").mkdir(parents=True, exist_ok=True)
    (export_dir / "model").mkdir(parents=True, exist_ok=True)
    (export_dir / "inference").mkdir(parents=True, exist_ok=True)

    class_index = {label: idx for idx, label in enumerate(labels)}
    class_index_path = export_dir / "data/class_index.json"
    class_index_path.write_text(json.dumps(class_index, indent=2), encoding="utf-8")

    signature = {
        "img_height": img_height,
        "img_width": img_width,
        "color_mode": "rgb",
        "preprocess": "tf.keras.applications.efficientnet.preprocess_input",
        "class_index_path": str(class_index_path),
        "created_utc": int(time.time()),
    }
    signature_path = export_dir / "inference/signature.json"
    signature_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")

    model_path = export_dir / "model/model.keras"
    model.save(model_path, include_optimizer=False)

    return model_path


def upload_model_to_registry(model_path: Path, run_id: str):
    bucket = os.getenv("S3_MODELS_BUCKET")
    if not bucket:
        print("[info] S3_MODELS_BUCKET not configured, skipping registry upload.")
        return
    s3_client = create_s3_client()
    key = f"skin/models/{run_id}/model.keras"
    s3_client.upload_file(str(model_path), bucket, key)
    print(f"[info] Uploaded model to s3://{bucket}/{key}")


def run_training(args: argparse.Namespace):
    cfg = load_dataset_config(Path(args.dataset_config))
    dataset_name = cfg.dataset_name

    records_train = fetch_records(cfg.metadata_db, dataset_name, args.train_splits)
    records_val = fetch_records(cfg.metadata_db, dataset_name, args.val_splits)
    if not records_train or not records_val:
        raise RuntimeError("No records found for training or validation splits. Run ingestion first.")

    label_to_index = {label: idx for idx, label in enumerate(cfg.labels)}

    cache_dir = Path(args.cache_dir)
    s3_client = create_s3_client()
    download_fn = make_download_fn(s3_client)

    def prepare(records):
        local_paths = []
        y = []
        for record in records:
            local_path = ensure_local_file(record, cache_dir, download_fn)
            local_paths.append(str(local_path))
            y.append(label_to_index[record["label"]])
        return local_paths, y

    train_paths, train_labels = prepare(records_train)
    val_paths, val_labels = prepare(records_val)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    experiment_name = args.experiment or f"{dataset_name}-training"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_params(
            {
                "dataset": dataset_name,
                "train_split": ",".join(args.train_splits),
                "val_split": ",".join(args.val_splits),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "img_height": args.img_height,
                "img_width": args.img_width,
                "learning_rate": args.learning_rate,
                "cache_dir": str(cache_dir),
            }
        )

        train_ds = build_dataset(
            train_paths,
            train_labels,
            args.img_height,
            args.img_width,
            args.batch_size,
            shuffle=True,
        )
        val_ds = build_dataset(
            val_paths,
            val_labels,
            args.img_height,
            args.img_width,
            args.batch_size,
            shuffle=False,
        )

        model = build_model(args.img_height, args.img_width, len(cfg.labels))
        optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        callbacks = [
            keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_accuracy"),
            MLflowLoggingCallback(),
        ]

        history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks, verbose=1)

        val_probs = model.predict(val_ds, verbose=0)
        val_pred = np.argmax(val_probs, axis=1)
        val_true = np.array(val_labels)
        val_f1 = float(f1_score(val_true, val_pred, average="macro"))

        val_probs = model.predict(val_ds, verbose=0)
        val_pred = np.argmax(val_probs, axis=1)
        val_true = np.array(val_labels)
        val_f1 = float(f1_score(val_true, val_pred, average="macro"))

        train_acc = float(history.history["accuracy"][-1])
        val_acc = float(history.history["val_accuracy"][-1])
        val_loss = float(history.history["val_loss"][-1])

        mlflow.log_metric("train_accuracy", train_acc, step=args.epochs)
        mlflow.log_metric("val_accuracy", val_acc, step=args.epochs)
        mlflow.log_metric("val_loss", val_loss, step=args.epochs)
        mlflow.log_metric("val_f1_macro", val_f1, step=args.epochs)

        export_dir = Path(args.export_dir)
        model_path = export_artifacts(export_dir, cfg.labels, args.img_height, args.img_width, model)

        mlflow.log_artifact(model_path, artifact_path="model")
        mlflow.log_artifact(model_path.parent.parent / "data" / "class_index.json", artifact_path="data")
        mlflow.log_artifact(model_path.parent.parent / "inference" / "signature.json", artifact_path="inference")

        upload_model_to_registry(model_path, run.info.run_id)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EfficientNet model with MLflow tracking")
    parser.add_argument("--dataset-config", type=Path, default=Path("src/data/dataset_config.yaml"))
    parser.add_argument("--train-splits", nargs="+", default=["train"])
    parser.add_argument("--val-splits", nargs="+", default=["val"])
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/dataset"))
    parser.add_argument("--export-dir", type=Path, default=Path("runs/skin5-current"))
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--run-name", type=str, default="skin-disease-train")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-height", type=int, default=224)
    parser.add_argument("--img-width", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
