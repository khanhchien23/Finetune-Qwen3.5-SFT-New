#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run Air-only Qwen3.5-9B LoRA SFT on Modal.

The function downloads the Air-only Parquet dataset from Hugging Face into a
Modal Volume, then runs the existing SFT trainer. It does not load GRPO or
SFT-2004 adapters: this is a fresh LoRA SFT run from Qwen/Qwen3.5-9B.

Example from CMD/PowerShell:
    modal run --detach modal_sft_air_only.py --hf-token hf_xxx \
      --hub-repo KhanhChien/FineTuneQwen35-9B-Air-Only
"""

import os
import subprocess
import sys
from pathlib import Path

import modal


APP_NAME = "qwen35-9b-sft-air-only"
DATASET_DEFAULT = "KhanhChien/Data-Qwen3.5-SFT-Air-Only"
OUTPUT_DEFAULT = "/data/outputs/Qwen3.5-9B-SFT-Air-Only"

sft_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "git-lfs")
    .pip_install(
        "torch>=2.6.0",
        "torchvision",
        index_url="https://download.pytorch.org/whl/cu126",
    )
    .pip_install(
        "transformers>=4.57.0",
        "accelerate>=1.2.0",
        "peft>=0.14.0",
        "qwen-vl-utils",
        "datasets",
        "pyarrow",
        "huggingface_hub>=0.24",
        "pillow",
        "tqdm",
        "tensorboard",
        "wandb",
        "flash-linear-attention",
    )
    .add_local_file(
        Path(__file__).parent / "train_sft_qwen4b_81classes.py",
        remote_path="/root/train_sft_qwen4b_81classes.py",
    )
)

volume = modal.Volume.from_name("qwen9b-sft-air-only-volume", create_if_missing=True)
app = modal.App(APP_NAME)


@app.function(
    image=sft_image,
    gpu="H100",
    timeout=24 * 3600,
    volumes={"/data": volume},
    cpu=12.0,
    memory=65536,
)
def run_sft_air_only(
    hf_token: str,
    dataset_repo: str = DATASET_DEFAULT,
    hub_repo: str = "",
    model_id: str = "Qwen/Qwen3.5-9B",
    epochs: float = 1.0,
    batch_size: int = 8,
    grad_accum: int = 4,
    lr: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    save_steps: int = 100,
    max_seq_length: int = 1280,
    resume: bool = True,
):
    from huggingface_hub import snapshot_download

    os.environ["HF_HOME"] = "/data/hf_cache"
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    dataset_dir = Path("/data/datasets") / dataset_repo.replace("/", "__")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    parquet_files = list(dataset_dir.rglob("*.parquet"))
    if not parquet_files:
        print(f"--> Downloading dataset {dataset_repo} -> {dataset_dir}", flush=True)
        snapshot_download(
            repo_id=dataset_repo,
            repo_type="dataset",
            local_dir=str(dataset_dir),
            token=hf_token,
        )
        volume.commit()
    else:
        print(f"--> Dataset already cached: {len(parquet_files)} Parquet files", flush=True)

    train_files = list(dataset_dir.rglob("train*.parquet"))
    if not train_files:
        raise FileNotFoundError(f"No train*.parquet found in {dataset_dir}")
    validation_files = list(dataset_dir.rglob("validation*.parquet"))
    print(f"--> Air-only dataset: train={len(train_files)} file(s), "
          f"validation={len(validation_files)} file(s)", flush=True)

    output_dir = Path(OUTPUT_DEFAULT)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_script = Path("/root/train_sft_qwen4b_81classes.py")
    cmd = [
        sys.executable, str(train_script),
        "--model_id", model_id,
        "--parquet_dir", str(dataset_dir),
        "--output_dir", str(output_dir),
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--grad_accum", str(grad_accum),
        "--lr", str(lr),
        "--lora_r", str(lora_r),
        "--lora_alpha", str(lora_alpha),
        "--save_steps", str(save_steps),
        "--max_seq_length", str(max_seq_length),
    ]
    if hub_repo:
        cmd += ["--hub_repo", hub_repo]
    if not resume:
        # Use a clean output directory when explicitly requested.
        output_dir = Path(f"/data/outputs/Qwen3.5-9B-SFT-Air-Only-fresh")
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd[cmd.index("--output_dir") + 1] = str(output_dir)

    print("--> Starting Air-only SFT:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    volume.commit()
    print(f"--> SFT finished. Output persisted at {output_dir}", flush=True)


@app.local_entrypoint()
def main(
    hf_token: str = "",
    dataset_repo: str = DATASET_DEFAULT,
    hub_repo: str = "KhanhChien/FineTuneQwen35-9B-Air-Only",
    model_id: str = "Qwen/Qwen3.5-9B",
    epochs: float = 1.0,
    batch_size: int = 8,
    grad_accum: int = 4,
    lr: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    save_steps: int = 100,
    max_seq_length: int = 1280,
    resume: bool = True,
):
    token = hf_token or os.environ.get("HF_TOKEN", "")
    if not token:
        raise ValueError("Pass --hf-token hf_xxx or set HF_TOKEN. It needs read/write access.")
    run_sft_air_only.remote(
        hf_token=token,
        dataset_repo=dataset_repo,
        hub_repo=hub_repo,
        model_id=model_id,
        epochs=epochs,
        batch_size=batch_size,
        grad_accum=grad_accum,
        lr=lr,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        save_steps=save_steps,
        max_seq_length=max_seq_length,
        resume=resume,
    )