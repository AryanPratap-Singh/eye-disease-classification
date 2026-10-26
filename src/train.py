import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
from .common import set_seed, ensure_dirs, MODEL_PATH, METADATA_PATH, REPORTS, IMAGE_SIZE, SEED, write_json
from .data import make_datasets
from .model import build_model


def main():
    parser = argparse.ArgumentParser(description="Train an EfficientNetB0 retinal image classifier.")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--fine-tune", action="store_true", help="Fine-tune the final backbone layers after head training.")
    parser.add_argument("--resume", action="store_true", help="Resume from the existing checkpoint when compatible.")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    set_seed(args.seed); ensure_dirs()
    train, val, test, classes = make_datasets(args.data_dir, args.image_size, args.batch_size, args.seed)
    print(f"Classes ({len(classes)}): {classes}")
    model = build_model(len(classes), args.image_size, fine_tune=args.fine_tune)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(MODEL_PATH), monitor="val_loss", save_best_only=True,
                                           verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
        tf.keras.callbacks.CSVLogger(str(REPORTS / "training_log.csv")),
    ]
    if args.resume and MODEL_PATH.exists():
        model = tf.keras.models.load_model(MODEL_PATH)
    history = model.fit(train, validation_data=val, epochs=args.epochs, callbacks=callbacks)
    model = tf.keras.models.load_model(MODEL_PATH)
    split_counts = {}
    from .data import validate_split
    for split in ("train", "val", "test"):
        split_counts[split] = validate_split(Path(args.data_dir) / split, classes)
    metadata = {"class_names": classes, "image_size": args.image_size, "seed": args.seed,
                "architecture": "EfficientNetB0 transfer learning", "dataset_dir": str(Path(args.data_dir).resolve()),
                "split_counts": split_counts, "fine_tune": args.fine_tune}
    write_json(METADATA_PATH, metadata)
    write_json(REPORTS / "training_history.json", history.history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history.get("accuracy", []), label="Train")
    axes[0].plot(history.history.get("val_accuracy", []), label="Validation")
    axes[0].set_title("Accuracy"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].plot(history.history.get("loss", []), label="Train")
    axes[1].plot(history.history.get("val_loss", []), label="Validation")
    axes[1].set_title("Loss"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    fig.tight_layout(); fig.savefig(REPORTS / "training_curves.png", dpi=160); plt.close(fig)
    print(f"Saved best model: {MODEL_PATH}\nSaved metadata: {METADATA_PATH}")
    print("Next: python -m src.evaluate --data-dir " + str(Path(args.data_dir) / "test"))

if __name__ == "__main__": main()
