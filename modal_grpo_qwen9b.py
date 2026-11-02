#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script chạy huấn luyện GRPO cho Qwen3.5-9B (`checkpoint-2004`) trên nền tảng đám mây MODAL (A100-80GB / H100).

Cách chạy trên máy tính Windows của bạn:
  1. Cài đặt Modal:
       pip install modal
  2. Đăng nhập Modal:
       modal setup
  3. Chạy huấn luyện GRPO trên đám mây Modal (có `--detach` để tắt máy tính vẫn tự chạy và đẩy lên HuggingFace):
       modal run --detach modal_grpo_qwen9b.py --max-steps 300 --save-steps 50
"""

import os
from pathlib import Path
import modal

# ==============================================================================
# 1. CẤU HÌNH CONTAINER IMAGE & VOLUME LƯU TRỮ BỀN VỮNG TRÊN MODAL
# ==============================================================================
APP_NAME = "qwen35-9b-grpo-ckpt2004"

# Tạo Image cài sẵn PyTorch 2.6 + CUDA 12.6 và các thư viện VLM cần thiết
grpo_image = (
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
    # Đóng gói trực tiếp file train_grpo_qwen9b_81classes.py vào trong container
    .add_local_file(
        Path(__file__).parent / "train_grpo_qwen9b_81classes.py",
        remote_path="/root/train_grpo_qwen9b_81classes.py",
    )
)

# Tạo Volume bền vững trên Modal để lưu cache HuggingFace (model 9B + dataset Parquet)
# và lưu các checkpoint GRPO (nếu chạy lại không bao giờ phải tải lại 18GB model)
vol = modal.Volume.from_name("qwen9b-grpo-volume", create_if_missing=True)

app = modal.App(APP_NAME)


# ==============================================================================
# 2. HÀM CHẠY TRÊN GPU H100 (80GB HBM3 - SIÊU TỐC) CỦA MODAL
# ==============================================================================
@app.function(
    image=grpo_image,
    gpu="H100",               # H100 80GB HBM3 (3350 GB/s) + batch_size=4 x G=4 (16 chuỗi/lượt)
    timeout=24 * 3600,        # Tối đa 24 tiếng để chạy hết toàn bộ bộ dữ liệu
    volumes={"/data": vol},   # Gắn ổ cứng bền vững vào /data
    cpu=8.0,
    memory=32768,
)
def run_grpo_on_modal(
    hf_token: str,
    wandb_key: str = "",
    sft_repo: str = "KhanhChien/FineTuneQwen35-9B-New",
    sft_checkpoint: str = "checkpoint-2004",
    dataset_repo: str = "KhanhChien/Data-Qwen3.5-SFT-New",
    hub_repo: str = "KhanhChien/GRPO-Qwen35-9B-New",
    epochs: int = 1,
    max_steps: int = 0,
    save_steps: int = 100,
    batch_size: int = 4,
    num_generations: int = 4,
    grad_accum: int = 8,
    lr: float = 5e-6,
    kl_beta: float = 0.02,
):
    import subprocess
    import sys

    # Đặt cache HuggingFace vào ổ Volume /data để tái sử dụng siêu tốc giữa các lần chạy
    from huggingface_hub import snapshot_download, hf_hub_download

    os.environ["HF_HOME"] = "/data/hf_cache"
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    if wandb_key:
        os.environ["WANDB_API_KEY"] = wandb_key

    # 1. Tự động clone / pull repo code từ GitHub về ổ đĩa Modal (/data/code)
    github_repo = "https://github.com/khanhchien23/Finetune-Qwen3.5-SFT-New.git"
    code_dir = Path("/data/code/Finetune-Qwen3.5-SFT-New")
    code_dir.parent.mkdir(parents=True, exist_ok=True)
    if (code_dir / ".git").is_dir():
        print(f"--> [1/4] Đang git pull repo code tại {code_dir}...", flush=True)
        subprocess.run(["git", "-C", str(code_dir), "fetch", "--all"], check=True)
        subprocess.run(["git", "-C", str(code_dir), "reset", "--hard", "origin/main"], check=True)
    else:
        print(f"--> [1/4] Đang git clone {github_repo} -> {code_dir}...", flush=True)
        subprocess.run(["git", "clone", github_repo, str(code_dir)], check=True)

    train_script = code_dir / "train_grpo_qwen9b_81classes.py"
    if not train_script.is_file():
        train_script = Path("/root/train_grpo_qwen9b_81classes.py")

    # 2. Tự động clone / tải toàn bộ repo Dataset Parquet về ổ đĩa Modal (/data/datasets)
    local_dataset_dir = Path("/data/datasets/Data-Qwen3.5-SFT-New")
    local_dataset_dir.mkdir(parents=True, exist_ok=True)
    if not list(local_dataset_dir.rglob("*.parquet")):
        print(f"--> [2/4] Đang clone/tải toàn bộ Dataset ({dataset_repo}) về {local_dataset_dir}...", flush=True)
        snapshot_download(
            repo_id=dataset_repo,
            repo_type="dataset",
            local_dir=str(local_dataset_dir),
            token=hf_token,
        )
        vol.commit()
    else:
        print(f"--> [2/4] Đã có sẵn Dataset Parquet trong ổ đĩa Modal: {local_dataset_dir}", flush=True)

    # 3. Tự động tải SFT checkpoint-2004 về ổ đĩa Modal (/data/sft_checkpoints)
    sft_root_dir = Path("/data/sft_checkpoints")
    sft_ckpt_dir = sft_root_dir / sft_checkpoint
    sft_ckpt_dir.mkdir(parents=True, exist_ok=True)
    weights_file = sft_ckpt_dir / "adapter_model.safetensors"
    if not (weights_file.is_file() and weights_file.stat().st_size > 1_000_000):
        print(f"--> [3/4] Đang tải {sft_checkpoint} từ {sft_repo} về {sft_ckpt_dir}...", flush=True)
        for fname in ("adapter_config.json", "adapter_model.safetensors"):
            hf_hub_download(
                repo_id=sft_repo,
                filename=f"{sft_checkpoint}/{fname}",
                local_dir=str(sft_root_dir),
                force_download=True,
                token=hf_token,
            )
        vol.commit()
    else:
        print(f"--> [3/4] Đã có sẵn {sft_checkpoint} tại {sft_ckpt_dir}", flush=True)

    # 4. Chạy huấn luyện GRPO trực tiếp trên GPU của Modal
    output_dir = f"/data/outputs/Qwen3.5-9B-GRPO-{sft_checkpoint}"
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

    print("--> [4/4] Bắt đầu chạy lệnh GRPO trên Modal GPU:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    vol.commit()
    print("--> Đã commit dữ liệu vào Modal Volume và đẩy toàn bộ checkpoint lên HuggingFace!", flush=True)


# ==============================================================================
# 3. LOCAL ENTRYPOINT (CHẠY TỪ TERMINAL MÁY WINDOWS CỦA BẠN)
# ==============================================================================
@app.local_entrypoint()
def main(
    hf_token: str = "",
    wandb_key: str = "",
    sft_repo: str = "KhanhChien/FineTuneQwen35-9B-New",
    sft_checkpoint: str = "checkpoint-2004",
    dataset_repo: str = "KhanhChien/Data-Qwen3.5-SFT-New",
    hub_repo: str = "KhanhChien/GRPO-Qwen35-9B-New",
    epochs: int = 1,
    max_steps: int = 0,
    save_steps: int = 100,
    batch_size: int = 4,
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
            "  modal run --detach modal_grpo_qwen9b.py --hf-token hf_xxxxxx --wandb-key xxxx"
        )
    run_grpo_on_modal.remote(
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
