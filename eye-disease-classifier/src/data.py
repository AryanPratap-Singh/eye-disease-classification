"""Dataset discovery, validation, duplicate checks, and TensorFlow datasets."""
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Iterable

SUPPORTED = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def discover_classes(split_dir: str | Path) -> list[str]:
    path = Path(split_dir)
    if not path.is_dir(): raise FileNotFoundError(f"Dataset split not found: {path}")
    classes = sorted(p.name for p in path.iterdir() if p.is_dir() and not p.name.startswith("."))
    if len(classes) < 2: raise ValueError(f"Expected at least two class folders in {path}; found {len(classes)}")
    return classes

def image_files(folder: str | Path) -> list[Path]:
    return sorted(p for p in Path(folder).rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)

def validate_split(split_dir: str | Path, expected_classes: Iterable[str] | None = None) -> dict[str, int]:
    path = Path(split_dir); classes = discover_classes(path)
    if expected_classes is not None and classes != sorted(expected_classes):
        raise ValueError(f"Class folders in {path} are {classes}, expected {sorted(expected_classes)}")
    counts = {name: len(image_files(path / name)) for name in classes}
    empty = [name for name, count in counts.items() if count == 0]
    if empty: raise ValueError(f"These class folders contain no supported images: {empty}")
    return counts

def exact_duplicate_groups(root: str | Path) -> list[list[str]]:
    """Return groups of files with identical bytes; useful before splitting."""
    hashes: dict[str, list[str]] = {}
    for path in image_files(root):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes.setdefault(digest, []).append(str(path))
    return [paths for paths in hashes.values() if len(paths) > 1]

def make_datasets(data_dir: str | Path, image_size: int = 224, batch_size: int = 16, seed: int = 42):
    import tensorflow as tf
    root = Path(data_dir); classes = discover_classes(root / "train")
    for name in ("train", "val", "test"): validate_split(root / name, classes)
    common = dict(image_size=(image_size, image_size), batch_size=batch_size, label_mode="int")
    train = tf.keras.utils.image_dataset_from_directory(root / "train", shuffle=True, seed=seed, **common)
    val = tf.keras.utils.image_dataset_from_directory(root / "val", shuffle=False, **common)
    test = tf.keras.utils.image_dataset_from_directory(root / "test", shuffle=False, **common)
    if train.class_names != classes or val.class_names != classes or test.class_names != classes:
        raise ValueError("Class mapping differs between dataset splits.")
    augmenter = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"), tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.10)], name="train_augmentation")
    train = train.map(lambda x, y: (augmenter(tf.cast(x, tf.float32), training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    val = val.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE)
    test = test.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE)
    return tuple(ds.prefetch(tf.data.AUTOTUNE) for ds in (train, val, test)) + (classes,)
