#!/usr/bin/env bash
# ==============================================================================
#   ONE-SHOT PIPELINE CHẠY GRPO TRỰC TIẾP TRÊN MODAL NOTEBOOK / TERMINAL / VAST.AI
#   Tự động:
#     1. Cài đặt thư viện cần thiết
#     2. Tải Dataset Parquet `KhanhChien/Data-Qwen3.5-SFT-New` về ổ đĩa local
#     3. Tải SFT `checkpoint-2004` từ `KhanhChien/FineTuneQwen35-9B-New`
#     4. Chạy `train_grpo_qwen9b_81classes.py` và đẩy các checkpoint GRPO lên HF Hub
# ==============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(pwd)"

HF_DATASET_REPO="${HF_DATASET_REPO:-KhanhChien/Data-Qwen3.5-SFT-New}"
LOCAL_DATA_DIR="${LOCAL_DATA_DIR:-${SCRIPT_DIR}/data_hf/Data-Qwen3.5-SFT-New}"

SFT_REPO="${SFT_REPO:-KhanhChien/FineTuneQwen35-9B-New}"
SFT_CHECKPOINT="${SFT_CHECKPOINT:-checkpoint-2004}"
HF_PUSH_REPO="${HF_PUSH_REPO:-KhanhChien/GRPO-Qwen35-9B-New}"

MAX_STEPS="${MAX_STEPS:-300}"
SAVE_STEPS="${SAVE_STEPS:-50}"
NUM_GEN="${NUM_GEN:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-5e-6}"
KL_BETA="${KL_BETA:-0.02}"

echo "[1/3] Cài đặt các thư viện cần thiết..."
pip install -q -U "transformers>=4.57.0" "accelerate>=1.2.0" "peft>=0.14.0" \
    qwen-vl-utils datasets pyarrow "huggingface_hub[hf_transfer]" pillow tqdm tensorboard wandb

echo "[2/3] Tải repo Dataset Parquet (${HF_DATASET_REPO}) về ${LOCAL_DATA_DIR}..."
python3 - <<PYEOF
import os
from pathlib import Path
from huggingface_hub import snapshot_download

data_dir = Path("${LOCAL_DATA_DIR}")
data_dir.mkdir(parents=True, exist_ok=True)
parquets = list(data_dir.rglob("*.parquet"))
if parquets:
    print(f"  -> Đã có sẵn {len(parquets)} file Parquet tại {data_dir}")
else:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    snapshot_download(
        repo_id="${HF_DATASET_REPO}",
        repo_type="dataset",
        local_dir=str(data_dir),
    )
    print(f"  -> Đã tải xong Dataset về {data_dir}")
PYEOF

echo "[3/3] Bắt đầu huấn luyện GRPO từ ${SFT_CHECKPOINT}..."
python3 "${SCRIPT_DIR}/train_grpo_qwen9b_81classes.py" \
    --sft_repo "${SFT_REPO}" \
    --sft_checkpoint "${SFT_CHECKPOINT}" \
    --dataset "${LOCAL_DATA_DIR}" \
    --output_dir "${SCRIPT_DIR}/Qwen3.5-9B-GRPO-${SFT_CHECKPOINT}" \
    --hub_repo "${HF_PUSH_REPO}" \
    --max_steps "${MAX_STEPS}" \
    --save_steps "${SAVE_STEPS}" \
    --num_generations "${NUM_GEN}" \
    --grad_accum "${GRAD_ACCUM}" \
    --lr "${LR}" \
    --kl_beta "${KL_BETA}"
