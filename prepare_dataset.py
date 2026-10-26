"""Prepare class-folder images into train/validation/test directories.

Example:
  python prepare_dataset.py --source "data/raw/Image Dataset on Eye Diseases Classification (Uveitis, Conjunctivitis, Cataract, Eyelid) with Symptoms and SMOTE Validation" --output data/processed

This is an image-level split. Prefer patient/source-level splitting when IDs are available.
"""
import argparse, random, shutil, hashlib, json
from pathlib import Path
from PIL import Image

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def readable_images(folder):
    result = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTS:
            try:
                with Image.open(p) as im:
                    im.verify()
                result.append(p)
            except Exception:
                print(f"Skipping unreadable image: {p}")
    return sorted(result)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Directory containing one folder per class")
    parser.add_argument("--output", default="data/processed")
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.train <= 0 or args.val <= 0 or args.train + args.val >= 1:
        parser.error("train and val must be positive and sum to less than 1")
    source = Path(args.source)
    if not source.is_dir():
        raise SystemExit(f"Source folder not found: {source}")
    classes = sorted(p for p in source.iterdir() if p.is_dir() and not p.name.startswith("."))
    if len(classes) < 2:
        raise SystemExit("Source must contain class subfolders, such as Cataract, Eyelid, etc.")
    rng = random.Random(args.seed)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Output directory is not empty: {output}. Choose a new directory or remove the old prepared split deliberately.")
    source_hashes = {}
    for cls in classes:
        for image in readable_images(cls):
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            source_hashes.setdefault(digest, []).append(str(image))
    duplicate_groups = [paths for paths in source_hashes.values() if len(paths) > 1]
    if duplicate_groups:
        print(f"Warning: {len(duplicate_groups)} exact duplicate groups found in source; keeping identical files in one source group is recommended.")
    manifest = {"seed": args.seed, "ratios": {"train": args.train, "val": args.val, "test": 1-args.train-args.val}, "classes": [p.name for p in classes], "files": []}
    for cls in classes:
        images = readable_images(cls)
        if len(images) < 5:
            raise SystemExit(f"Class {cls.name} has only {len(images)} readable images; need at least 5.")
        rng.shuffle(images)
        n_train = int(len(images) * args.train)
        n_val = int(len(images) * args.val)
        split_map = {
            "train": images[:n_train],
            "val": images[n_train:n_train+n_val],
            "test": images[n_train+n_val:],
        }
        for split, paths in split_map.items():
            dest = output / split / cls.name
            dest.mkdir(parents=True, exist_ok=True)
            for idx, src in enumerate(paths):
                shutil.copy2(src, dest / f"{idx:05d}_{src.name}")
                manifest["files"].append({"source": str(src), "split": split, "class": cls.name, "sha256": hashlib.sha256(src.read_bytes()).hexdigest()})
        print(f"{cls.name}: " + ", ".join(f"{s}={len(ps)}" for s, ps in split_map.items()))
    print(f"Prepared data at {output.resolve()}")
    print("Warning: image-level splitting can leak duplicates or augmented siblings across splits.")
    print("Inspect duplicate/augmented images and use patient-level splits if patient IDs are available.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
