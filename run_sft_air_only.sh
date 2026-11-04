#!/usr/bin/env bash
set -euo pipefail

# SFT LoRA Qwen3.5-9B on the Air-only Parquet dataset.
# The default source is a Hugging Face Dataset repo; it is downloaded to a
# local cache/container before training. Override DATASET_REPO, DATA_DIR,
# OUTPUT_DIR, HUB_REPO and training parameters with env vars.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_REPO="${DATASET_REPO:-KhanhChien/Data-Qwen3.5-SFT-Air-Only}"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/../hf_sft_air_only_parquet}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${SCRIPT_DIR}/../hf_cache_air_only}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/Qwen3.5-9B-SFT-Air-Only}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-9B}"
HUB_REPO="${HUB_REPO:-}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-2e-4}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
SAVE_STEPS="${SAVE_STEPS:-100}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-1280}"

if [[ ! -d "${DATA_DIR}" || -z "$(find "${DATA_DIR}/data" -name '*.parquet' -print -quit 2>/dev/null)" ]]; then
    echo "Downloading Air-only dataset ${DATASET_REPO} -> ${DATA_DIR}"
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "[ERROR] HF_TOKEN is required to download a private dataset." >&2
        exit 1
    fi
    python3 -m pip install -q -U "huggingface_hub>=0.24"
    python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="${DATASET_REPO}", repo_type="dataset", local_dir="${DATA_DIR}",
                  cache_dir="${HF_CACHE_DIR}", token="${HF_TOKEN}")
PY
fi

ARGS=(
    "${SCRIPT_DIR}/train_sft_qwen4b_81classes.py"
    --model_id "${MODEL_ID}"
    --parquet_dir "${DATA_DIR}"
    --output_dir "${OUTPUT_DIR}"
    --epochs "${EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --grad_accum "${GRAD_ACCUM}"
    --lr "${LR}"
    --lora_r "${LORA_R}"
    --lora_alpha "${LORA_ALPHA}"
    --save_steps "${SAVE_STEPS}"
    --max_seq_length "${MAX_SEQ_LENGTH}"
)
if [[ -n "${HUB_REPO}" ]]; then
    ARGS+=(--hub_repo "${HUB_REPO}")
fi

echo "Starting Air-only SFT"
echo "  dataset : ${DATA_DIR}"
echo "  dataset repo: ${DATASET_REPO}"
echo "  output  : ${OUTPUT_DIR}"
echo "  model   : ${MODEL_ID}"
python3 "${ARGS[@]}"