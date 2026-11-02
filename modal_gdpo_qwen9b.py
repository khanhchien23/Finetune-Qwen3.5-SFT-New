#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script chạy huấn luyện GDPO (Group Decoupled Policy Optimization) cho Qwen3.5-9B (`checkpoint-2004`)
trực tiếp trên đám mây MODAL (A100-80GB / H100).

Cách chạy từ máy tính:
  modal run --detach modal_gdpo_qwen9b.py --hf-token hf_xxxxxx --max-steps 300 --save-steps 50
"""

import os
import subprocess
import sys
from pathlib import Path
import modal

APP_NAME = "qwen35-9b-gdpo-ckpt2004"

gdpo_image = (
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
        "huggingface_hub",
        "pillow",
        "tqdm",
        "tensorboard",
        "wandb",
        "flash-linear-attention",
    )
    .add_local_file(
        Path(__file__).parent / "train_gdpo_qwen9b_81classes.py",
        remote_path="/root/train_gdpo_qwen9b_81classes.py",
    )
)

vol = modal.Volume.from_name("qwen9b-grpo-volume", create_if_missing=True)
app = modal.App(APP_NAME)


@app.function(
    image=gdpo_image,
    gpu="H100",               # H100 80GB HBM3 (3350 GB/s) + batch_size=4 x G=4 (16 chuỗi/lượt)
    timeout=24 * 3600,        # Tối đa 24 tiếng để chạy hết toàn bộ bộ dữ liệu
    volumes={"/data": vol},
    cpu=8.0,
    memory=32768,
)
def run_gdpo_on_modal(
    hf_token: str,
    wandb_key: str = "",
    sft_repo: str = "KhanhChien/FineTuneQwen35-9B-New",
    sft_checkpoint: str = "checkpoint-2004",
    dataset_repo: str = "KhanhChien/Data-Qwen3.5-SFT-New",
    hub_repo: str = "KhanhChien/GDPO-Qwen35-9B-New",
    epochs: int = 1,
    max_steps: int = 0,
    save_steps: int = 50,
    batch_size: int = 8,
    num_generations: int = 4,
    grad_accum: int = 8,
    lr: float = 5e-6,
    kl_beta: float = 0.02,
):
    from huggingface_hub import snapshot_download, hf_hub_download

    os.environ["HF_HOME"] = "/data/hf_cache"
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    if wandb_key:
        os.environ["WANDB_API_KEY"] = wandb_key

    github_repo = "https://github.com/khanhchien23/Finetune-Qwen3.5-SFT-New.git"
    code_dir = Path("/data/code/Finetune-Qwen3.5-SFT-New")
    code_dir.parent.mkdir(parents=True, exist_ok=True)
    if (code_dir / ".git").is_dir():
        subprocess.run(["git", "-C", str(code_dir), "fetch", "--all"], check=True)
        subprocess.run(["git", "-C", str(code_dir), "reset", "--hard", "origin/main"], check=True)
    else:
        subprocess.run(["git", "clone", github_repo, str(code_dir)], check=True)

    train_script = code_dir / "train_gdpo_qwen9b_81classes.py"
    if not train_script.is_file():
        train_script = Path("/root/train_gdpo_qwen9b_81classes.py")

    local_dataset_dir = Path("/data/datasets/Data-Qwen3.5-SFT-New")
    local_dataset_dir.mkdir(parents=True, exist_ok=True)
    if not list(local_dataset_dir.rglob("*.parquet")):
        snapshot_download(
            repo_id=dataset_repo,
            repo_type="dataset",
            local_dir=str(local_dataset_dir),
            token=hf_token,
        )
        vol.commit()

    sft_root_dir = Path("/data/sft_checkpoints")
    sft_ckpt_dir = sft_root_dir / sft_checkpoint
    sft_ckpt_dir.mkdir(parents=True, exist_ok=True)
    weights_file = sft_ckpt_dir / "adapter_model.safetensors"
    if not (weights_file.is_file() and weights_file.stat().st_size > 1_000_000):
        for fname in ("adapter_config.json", "adapter_model.safetensors"):
            hf_hub_download(
                repo_id=sft_repo,
                filename=f"{sft_checkpoint}/{fname}",
                local_dir=str(sft_root_dir),
                force_download=True,
                token=hf_token,
            )
        vol.commit()

    output_dir = f"/data/outputs/Qwen3.5-9B-GDPO-{sft_checkpoint}"
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable,
        str(train_script),
        "--sft_repo", str(sft_root_dir),
        "--sft_checkpoint", sft_checkpoint,
        "--dataset", str(local_dataset_dir),
        "--output_dir", output_dir,
        "--hub_repo", hub_repo,
        "--epochs", str(epochs),
        "--max_steps", str(max_steps),
        "--save_steps", str(save_steps),
        "--batch_size", str(batch_size),
        "--num_generations", str(num_generations),
        "--grad_accum", str(grad_accum),
        "--lr", str(lr),
        "--kl_beta", str(kl_beta),
    ]
    print("--> Bắt đầu chạy lệnh GDPO trên Modal GPU:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    vol.commit()


@app.local_entrypoint()
def main(
    hf_token: str = "",
    wandb_key: str = "",
    sft_repo: str = "KhanhChien/FineTuneQwen35-9B-New",
    sft_checkpoint: str = "checkpoint-2004",
    dataset_repo: str = "KhanhChien/Data-Qwen3.5-SFT-New",
    hub_repo: str = "KhanhChien/GDPO-Qwen35-9B-New",
    epochs: int = 1,
    max_steps: int = 0,
    save_steps: int = 50,
    batch_size: int = 8,
    num_generations: int = 4,
    grad_accum: int = 8,
    lr: float = 5e-6,
    kl_beta: float = 0.02,
):
    token = hf_token or os.environ.get("HF_TOKEN", "")
    w_key = wandb_key or os.environ.get("WANDB_API_KEY", "")
    if not token:
        raise ValueError(
            "Bạn cần truyền HuggingFace Token có quyền Write!\n"
            "Ví dụ:\n"
            "  modal run --detach modal_gdpo_qwen9b.py --hf-token hf_xxxxxx --wandb-key xxxx"
        )
    run_gdpo_on_modal.remote(
        hf_token=token,
        wandb_key=w_key,
        sft_repo=sft_repo,
        sft_checkpoint=sft_checkpoint,
        dataset_repo=dataset_repo,
        hub_repo=hub_repo,
        epochs=epochs,
        max_steps=max_steps,
        save_steps=save_steps,
        batch_size=batch_size,
        num_generations=num_generations,
        grad_accum=grad_accum,
        lr=lr,
        kl_beta=kl_beta,
    )
