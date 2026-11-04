#!/usr/bin/env python
"""Create an Air-only copy of the Qwen VLM Parquet dataset.

The source dataset contains embedded images and a ``gt_class`` column. This
script keeps every original column/prompt/image and filters only rows whose
ground-truth label starts with ``Air.``. The output layout is compatible with
the existing GRPO loader::

    output/data/train-air-only.parquet
    output/data/validation-air-only.parquet

Examples:
    python create_air_only_dataset.py \
      --input D:\\du_lieu_that\\hf_sft_81classes_vlm_parquet \
      --output D:\\du_lieu_that\\hf_sft_air_only_parquet
"""

import argparse
from pathlib import Path

from datasets import load_dataset


def parquet_files(root: Path, split: str):
    files = sorted(root.rglob("*.parquet"))
    selected = [p for p in files if split in p.name.lower()]
    if not selected:
        raise FileNotFoundError(f"No {split} Parquet files found below {root}")
    return selected


def filter_split(root: Path, output_dir: Path, split: str):
    files = parquet_files(root, split)
    dataset = load_dataset("parquet", data_files={split: [str(p) for p in files]})[split]
    if "gt_class" not in dataset.column_names:
        raise ValueError(f"{split} dataset has no gt_class column: {dataset.column_names}")
    air = dataset.filter(
        lambda row: isinstance(row["gt_class"], str) and row["gt_class"].startswith("Air."),
        desc=f"Filtering Air-only {split}",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{split}-air-only.parquet"
    air.to_parquet(str(output_file))
    print(f"{split}: {len(dataset):,} -> {len(air):,} rows | {output_file}")
    return len(air)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Source dataset directory containing Parquet files")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; source is never modified")
    args = parser.parse_args()
    if not args.input.is_dir():
        raise FileNotFoundError(args.input)
    if args.input.resolve() == args.output.resolve():
        raise ValueError("--output must be different from --input")
    output_data = args.output / "data"
    train_count = filter_split(args.input, output_data, "train")
    val_count = filter_split(args.input, output_data, "validation")
    print(f"Created Air-only dataset: {args.output.resolve()}")
    print(f"Total: train={train_count:,}, validation={val_count:,}, all={train_count + val_count:,}")


if __name__ == "__main__":
    main()