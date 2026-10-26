"""Evaluate a saved model on the held-out test split."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from .common import MODEL_PATH, METADATA_PATH, REPORTS, ensure_dirs, read_json, write_json
from .data import validate_split

def calculate_metrics(y_true, y_pred, class_names: list[str]) -> dict:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
    labels = list(range(len(class_names)))
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "weighted_precision": float(precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
            "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "weighted_recall": float(recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
            "n_test_images": int(len(y_true)), "class_names": class_names}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed/test")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(); ensure_dirs()
    if not MODEL_PATH.exists() or not METADATA_PATH.exists(): raise FileNotFoundError("Model or metadata missing; train first.")
    meta = read_json(METADATA_PATH); classes = meta["class_names"]; validate_split(args.data_dir, classes)
    import numpy as np
    import tensorflow as tf
    ds = tf.keras.utils.image_dataset_from_directory(args.data_dir, image_size=(meta["image_size"],)*2,
        batch_size=args.batch_size, label_mode="int", shuffle=False)
    if ds.class_names != classes: raise ValueError(f"Test class order {ds.class_names} differs from saved mapping {classes}")
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y)).prefetch(tf.data.AUTOTUNE)
    y_true = np.concatenate([y.numpy() for _, y in ds]); model = tf.keras.models.load_model(MODEL_PATH)
    probs = model.predict(ds, verbose=1); y_pred = probs.argmax(axis=1)
    from sklearn.metrics import classification_report, confusion_matrix
    metrics = calculate_metrics(y_true, y_pred, classes)
    report = classification_report(y_true, y_pred, labels=list(range(len(classes))), target_names=classes, output_dict=True, zero_division=0)
    write_json(REPORTS / "metrics.json", metrics); write_json(REPORTS / "classification_report.json", report)
    import matplotlib.pyplot as plt, seaborn as sns
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    fig, ax = plt.subplots(figsize=(max(7, len(classes)*1.2), max(6, len(classes))))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label"); ax.set_title("Held-out Test Confusion Matrix")
    plt.xticks(rotation=35, ha="right"); plt.tight_layout(); fig.savefig(REPORTS / "confusion_matrix.png", dpi=180); plt.close(fig)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__": main()
