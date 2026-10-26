"""Report image counts by split/class without modifying dataset files."""
import argparse
from pathlib import Path
from src.data import SUPPORTED


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed")
    args = parser.parse_args()
    root = Path(args.data_dir)
    if not root.is_dir():
        raise SystemExit(f"Dataset directory not found: {root}")
    total = 0
    for split in ("train", "val", "test"):
        split_path = root / split
        if not split_path.is_dir():
            print(f"{split}: MISSING")
            continue
        print(f"\\n{split.upper()}")
        split_total = 0
        for class_dir in sorted(p for p in split_path.iterdir() if p.is_dir()):
            count = sum(1 for f in class_dir.rglob("*") if f.is_file() and f.suffix.lower() in SUPPORTED)
            print(f"  {class_dir.name}: {count}")
            split_total += count
        print(f"  Total: {split_total}")
        total += split_total
    print(f"\\nTotal images across available splits: {total}")

if __name__ == "__main__":
    main()
