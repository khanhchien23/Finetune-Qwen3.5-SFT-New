#!/usr/bin/env python
"""Export the Air-only SFT Parquet rows as readable JSONL.

This exports the *training targets* exactly as stored in the dataset. It does
not run the model and does not generate new CoT. Images are represented by
their Parquet image metadata/path by default, so the JSONL remains readable.
"""

import argparse
import base64
import io
import json
from pathlib import Path

from datasets import Image as HFImage
from datasets import load_dataset


def text_from_content(content):
    """Extract text from either a plain content string or chat content list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", "")) for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content)


def get_texts(row):
    """Read the canonical direct columns or the optional messages column."""
    if all(key in row for key in ("system_prompt", "user_prompt", "assistant_cot")):
        return row["system_prompt"], row["user_prompt"], row["assistant_cot"]
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        raise ValueError("Row has neither system_prompt/user_prompt/assistant_cot nor 3+ messages")
    return (
        text_from_content(messages[0].get("content", "")),
        text_from_content(messages[1].get("content", "")),
        text_from_content(messages[2].get("content", "")),
    )


def image_info(image, include_base64=False):
    """Return compact image information, optionally including PNG base64."""
    if image is None:
        return None
    if isinstance(image, dict):
        result = {key: value for key, value in image.items() if key != "bytes"}
        raw = image.get("bytes")
        if include_base64 and raw:
            result["base64_png_or_source"] = base64.b64encode(raw).decode("ascii")
        return result
    result = {"type": type(image).__name__}
    if hasattr(image, "size"):
        result["size"] = list(image.size)
    if include_base64:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        result["base64_png"] = base64.b64encode(buffer.getvalue()).decode("ascii")
    return result


def export_split(dataset_dir, split, output, limit, include_base64):
    parquet_files = sorted(dataset_dir.rglob(f"{split}*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No {split}*.parquet found below {dataset_dir}")
    dataset = load_dataset("parquet", data_files={split: [str(p) for p in parquet_files]})[split]
    if "image" in dataset.column_names:
        # Decode image metadata only when requested; otherwise keep export light.
        dataset = dataset.cast_column("image", HFImage(decode=include_base64))
    count = min(len(dataset), limit) if limit else len(dataset)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for index in range(count):
            row = dataset[index]
            system_prompt, user_prompt, assistant_cot = get_texts(row)
            if not isinstance(assistant_cot, str):
                assistant_cot = str(assistant_cot)
            if not assistant_cot.startswith("<think>"):
                raise ValueError(f"Row {index} assistant_cot does not start with <think>")
            record = {
                "split": split,
                "row_index": index,
                "gt_class": row.get("gt_class", ""),
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "assistant_cot": assistant_cot,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_cot},
                ],
                "image": image_info(row.get("image"), include_base64),
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"{split}: exported {count:,}/{len(dataset):,} rows -> {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True,
                        help="Air-only dataset directory containing train*.parquet and validation*.parquet")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory; default is <dataset-dir>/jsonl")
    parser.add_argument("--limit", type=int, default=0, help="Rows per split; 0 exports all")
    parser.add_argument("--split", choices=("train", "validation", "both"), default="both")
    parser.add_argument("--include-image-base64", action="store_true",
                        help="Embed every image as base64 PNG; creates very large JSONL files")
    args = parser.parse_args()
    if not args.dataset_dir.is_dir():
        raise FileNotFoundError(args.dataset_dir)
    if args.limit < 0:
        raise ValueError("--limit must be >= 0")
    output_dir = args.output_dir or args.dataset_dir / "jsonl"
    splits = ("train", "validation") if args.split == "both" else (args.split,)
    for split in splits:
        export_split(dataset_dir=args.dataset_dir, split=split,
                     output=output_dir / f"{split}-air-only.jsonl",
                     limit=args.limit, include_base64=args.include_image_base64)


if __name__ == "__main__":
    main()