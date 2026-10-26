"""Shared paths, reproducibility helpers, and JSON utilities."""
from __future__ import annotations
import json, os, random
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SIZE = 224
SEED = 42
ARTIFACTS = PROJECT_ROOT / "artifacts"
REPORTS = PROJECT_ROOT / "reports"
MODEL_PATH = ARTIFACTS / "retinal_classifier.keras"
METADATA_PATH = ARTIFACTS / "metadata.json"

def set_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass

def ensure_dirs() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True); REPORTS.mkdir(parents=True, exist_ok=True)

def read_json(path: Path) -> dict[str, Any]:
    if not path.exists(): raise FileNotFoundError(f"Missing {path}. Train the model first.")
    return json.loads(path.read_text(encoding="utf-8"))

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")

def load_metadata() -> dict[str, Any]:
    return read_json(METADATA_PATH)
