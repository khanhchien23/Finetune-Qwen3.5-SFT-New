#!/usr/bin/env python
"""Upload the already-created Air-only Parquet dataset to Hugging Face.

The Parquet rows contain the embedded image plus ``system_prompt``,
``user_prompt`` and ``assistant_cot``. No image files are omitted from the
uploaded dataset.

Usage:
    $env:HF_TOKEN = "hf_..."
    python upload_air_only_dataset.py `
      --local-dir D:\\du_lieu_that\\hf_sft_air_only_parquet `
      --repo-id KhanhChien/Data-Qwen3.5-SFT-Air-Only
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--public", action="store_true", help="Make the dataset public; private by default")
    args = parser.parse_args()
    if not (args.local_dir / "data").is_dir():
        raise FileNotFoundError(f"Expected Parquet dataset directory: {args.local_dir / 'data'}")
    parquet_files = sorted((args.local_dir / "data").glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found under {args.local_dir / 'data'}")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Set HF_TOKEN to a Hugging Face token with write access before uploading.")
    api = HfApi(token=token)
    create_repo(repo_id=args.repo_id, repo_type="dataset", private=not args.public, exist_ok=True, token=token)
    upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(args.local_dir),
        path_in_repo=".",
        commit_message="Add Air-only images and CoT Parquet dataset",
        token=token,
    )
    print(f"Uploaded {len(parquet_files)} Parquet files to https://huggingface.co/datasets/{args.repo_id}")
    for path in parquet_files:
        print(f"  - {path.name}: {path.stat().st_size / 1024**2:.1f} MiB")


if __name__ == "__main__":
    main()