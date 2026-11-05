"""Pack class-folder real images into one portable Parquet file.

The input may be either ``selected_no_errors`` or its nested
``selected_no_errors/selected_no_errors`` directory. Each immediate class
folder becomes the ``ground_truth`` column and image bytes are stored in
``image_bytes`` so the Parquet can be downloaded and tested elsewhere.

Examples:
    python pack_selected_no_errors_parquet.py \
      --input D:\\du_lieu_that\\selected_no_errors \
      --output D:\\du_lieu_that\\selected_no_errors.parquet
"""

import argparse
import hashlib
from pathlib import Path

from PIL import Image
from datasets import Dataset

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def resolve_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    if any(p.is_dir() and any(x.is_file() and x.suffix.lower() in EXTENSIONS for x in p.iterdir())
           for p in path.iterdir()):
        return path
    nested = path / path.name
    if nested.is_dir():
        return resolve_root(nested)
    candidates = [p for p in path.iterdir() if p.is_dir()]
    if len(candidates) == 1:
        return resolve_root(candidates[0])
    raise ValueError(f"Could not find class folders directly below {path}")


def collect(root: Path):
    rows = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for image_path in sorted(p for p in class_dir.rglob("*")
                                 if p.is_file() and p.suffix.lower() in EXTENSIONS):
            with Image.open(image_path) as image:
                image.verify()
            data = image_path.read_bytes()
            rows.append({
                "file_name": image_path.relative_to(root).as_posix(),
                "ground_truth": class_dir.name,
                "image_bytes": data,
                "image_format": image_path.suffix.lower().lstrip("."),
                "byte_size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            })
    if not rows:
        raise ValueError(f"No supported images below {root}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = resolve_root(args.input)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = collect(root)
    Dataset.from_list(rows).to_parquet(str(output))
    classes = sorted({row["ground_truth"] for row in rows})
    print(f"Input root: {root}")
    print(f"Wrote: {output}")
    print(f"Images: {len(rows):,} | classes: {len(classes)} | size: {output.stat().st_size / 1024**2:.1f} MiB")


if __name__ == "__main__":
    main()