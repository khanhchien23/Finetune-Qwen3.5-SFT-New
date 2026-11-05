#!/usr/bin/env python
"""Run GRPO inference on Modal H100; the Windows machine never loads Qwen 9B.

First upload the local Parquet to the existing Modal Volume:
    modal volume put qwen9b-grpo-volume D:\\du_lieu_that\\selected_no_errors.parquet /realtest/selected_no_errors.parquet

Then run:
    modal run --detach modal_inference_qwen9b.py --hf-token hf_xxx \
      --checkpoint checkpoint-grpo-233 --batch-size 32
"""

import os
import subprocess
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
APP_NAME = "qwen35-9b-grpo-inference"
vol = modal.Volume.from_name("qwen9b-grpo-volume", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.6.0", "torchvision", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install(
        "transformers>=4.57.0", "accelerate>=1.2.0", "peft>=0.14.0",
        "qwen-vl-utils", "datasets", "pyarrow", "huggingface_hub", "pillow", "tqdm",
    )
    .add_local_file(ROOT / "inference_selected_no_errors.py", "/root/inference_selected_no_errors.py")
)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu="H100",
    timeout=24 * 3600,
    volumes={"/data": vol},
    cpu=8.0,
    memory=32768,
)
def run_inference(
    hf_token: str,
    checkpoint: str = "checkpoint-grpo-233",
    data_path: str = "/data/realtest/selected_no_errors.parquet",
    output_name: str = "epoch2_predictions.csv",
    batch_size: int = 32,
    max_new_tokens: int = 384,
    limit: int = 0,
):
    os.environ["HF_HOME"] = "/data/hf_cache"
    os.environ["HF_HUB_CACHE"] = "/data/hf_cache/hub"
    os.environ["TRANSFORMERS_CACHE"] = "/data/hf_cache/transformers"
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    data = Path(data_path)
    if not data.is_file():
        raise FileNotFoundError(
            f"Missing {data}. Upload it first with: "
            "modal volume put qwen9b-grpo-volume <local.parquet> /realtest/selected_no_errors.parquet"
        )
    output_dir = Path("/data/inference_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / output_name
    cmd = [
        sys.executable, "/root/inference_selected_no_errors.py",
        "--data", str(data),
        "--adapter", "KhanhChien/GRPO-Qwen35-9B-New",
        "--checkpoint", checkpoint,
        "--output", str(output),
        "--batch-size", str(batch_size),
        "--max-new-tokens", str(max_new_tokens),
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    print("Running on Modal H100:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    vol.commit()
    print(f"Saved to Volume: {output}", flush=True)


@app.local_entrypoint()
def main(
    hf_token: str = "",
    checkpoint: str = "checkpoint-grpo-233",
    data_path: str = "/data/realtest/selected_no_errors.parquet",
    output_name: str = "epoch2_predictions.csv",
    batch_size: int = 32,
    max_new_tokens: int = 384,
    limit: int = 0,
):
    token = hf_token or os.environ.get("HF_TOKEN", "")
    if not token:
        raise ValueError("Pass a new HF read token with --hf-token or set HF_TOKEN")
    if batch_size < 1 or max_new_tokens < 1 or limit < 0:
        raise ValueError("batch_size/max_new_tokens must be positive and limit >= 0")
    run_inference.remote(
        hf_token=token,
        checkpoint=checkpoint,
        data_path=data_path,
        output_name=output_name,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        limit=limit,
    )