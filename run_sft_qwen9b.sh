#!/usr/bin/env bash
# ==============================================================================
#   SFT CoT 81 MILITARY CLASSES - ONE-SHOT PIPELINE FOR LINUX (CUDA)
#   Step 1: Check/install PyTorch >= 2.6 (CUDA) + Transformers/PEFT packages
#   Step 2: Pull Parquet dataset from HuggingFace Hub (skipped if already local)
#   Step 3: Train SFT with enable_thinking=True (ThinkingDataCollator)
#
#   Usage:
#     bash run_sft_qwen9b.sh                                  # mac dinh BATCH_SIZE=2, EPOCHS=1
#     bash run_sft_qwen9b.sh --batch_size 4                   # chinh batch_size truc tiep qua tham so
#     bash run_sft_qwen9b.sh --batch_size 4 --grad_accum 4    # chinh batch_size + grad_accum
#     BATCH_SIZE=4 GRAD_ACCUM=4 bash run_sft_qwen9b.sh        # hoac qua bien moi truong
#     NUM_GPUS=2 bash run_sft_qwen9b.sh                       # DDP qua torchrun
#     TORCH_SOURCE=PYPI bash run_sft_qwen9b.sh                # torch tu PyPI mac dinh
#     EPOCHS=1 LR=1e-4 bash run_sft_qwen9b.sh                 # override hyper-param
#     HF_PUSH_REPO="" bash run_sft_qwen9b.sh                  # tat day checkpoint len HF
#     HF_REPO_PUBLIC=1 bash run_sft_qwen9b.sh                 # repo HF cong khai
# ==============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(pwd)"

# ----------------------------------------------------------------------------
# 1. CAU HINH THAM SO (override bang bien moi truong HOAC tham so dong lenh)
# ----------------------------------------------------------------------------
HF_DATASET_REPO="${HF_DATASET_REPO:-KhanhChien/Data-Qwen3.5-SFT-New}"
LOCAL_DATA_DIR="${LOCAL_DATA_DIR:-${SCRIPT_DIR}/data_hf/Data-Qwen3.5-SFT-New}"

# Chay 9B 16-bit (LoRA, khong quantize) -> can GPU >= 20 GB (script kiem tra va dung som).
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-9B}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/$(basename "${MODEL_ID}")-SFT-CoT-81Classes-thinking}"

# Repo HuggingFace de tu day moi checkpoint + final_adapter len.
# Tat bang cach:  HF_PUSH_REPO="" bash run_sft_qwen9b.sh
# Mo repo cong khai:  HF_REPO_PUBLIC=1 bash run_sft_qwen9b.sh
# Can token:  export HF_TOKEN=hf_xxx   (hoac: huggingface-cli login)
HF_PUSH_REPO="${HF_PUSH_REPO-KhanhChien/FineTuneQwen35-9B}"
HF_REPO_PUBLIC="${HF_REPO_PUBLIC:-0}"

EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-2e-4}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
SAVE_STEPS="${SAVE_STEPS:-200}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1280}"
NUM_GPUS="${NUM_GPUS:-1}"
TB_PORT="${TB_PORT:-6006}"
START_TB="${START_TB:-1}"
# TORCH_SOURCE chi dung khi chua co torch >= 2.6 + CUDA; GPU Blackwell nen cai cu130 truoc.
TORCH_SOURCE="${TORCH_SOURCE:-CUDA_12_6}"

# Ho tro truyen tham so truc tiep tu dong lenh (vd: bash run_sft_qwen9b.sh --batch_size 4)
while [ "$#" -gt 0 ]; do
    case "$1" in
        -b|--batch_size|--batch-size)
            BATCH_SIZE="$2"; shift 2 ;;
        --batch_size=*|--batch-size=*)
            BATCH_SIZE="${1#*=}"; shift 1 ;;
        -g|--grad_accum|--grad-accum)
            GRAD_ACCUM="$2"; shift 2 ;;
        --grad_accum=*|--grad-accum=*)
            GRAD_ACCUM="${1#*=}"; shift 1 ;;
        -e|--epochs)
            EPOCHS="$2"; shift 2 ;;
        --epochs=*)
            EPOCHS="${1#*=}"; shift 1 ;;
        --lr)
            LR="$2"; shift 2 ;;
        --lr=*)
            LR="${1#*=}"; shift 1 ;;
        --save_steps|--save-steps)
            SAVE_STEPS="$2"; shift 2 ;;
        --max_seq_len|--max_seq_length)
            MAX_SEQ_LEN="$2"; shift 2 ;;
        --num_gpus|--num-gpus)
            NUM_GPUS="$2"; shift 2 ;;
        --hub_repo|--hub-repo)
            HF_PUSH_REPO="$2"; shift 2 ;;
        [0-9]*)
            BATCH_SIZE="$1"; shift 1 ;;
        *)
            echo "[ERROR] Tham so khong hop le: $1"
            echo "        Vi du: bash run_sft_qwen9b.sh --batch_size 4 --grad_accum 4 --epochs 1"
            exit 1 ;;
    esac
done

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export PYTHONUNBUFFERED=1

if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi

# Ubuntu/Debian ban minimal (dac biet la WSL) thuong chua co pip -> tu cai
if ! "${PY}" -m pip --version >/dev/null 2>&1; then
    echo "  -> Python pip khong co. Dang cai dat (apt / ensurepip)..."
    if command -v apt-get >/dev/null 2>&1; then
        APT_CMD=(apt-get)
        if [ "$(id -u)" -ne 0 ]; then
            if command -v sudo >/dev/null 2>&1; then
                APT_CMD=(sudo apt-get)
            else
                echo "[WARN] Chua co sudo va dang khong phai root -> khong tu cai apt duoc."
            fi
        fi
        if [ "${#APT_CMD[@]}" -gt 0 ]; then
            "${APT_CMD[@]}" update -qq || echo "  -> [WARN] apt-get update that bai (mang/guong?), van thu ensurepip..."
            "${APT_CMD[@]}" install -y -qq python3-pip python3-venv || true
        fi
    fi
    "${PY}" -m ensurepip --upgrade >/dev/null 2>&1 || true
    if ! "${PY}" -m pip --version >/dev/null 2>&1; then
        echo "[ERROR] Van chua cai duoc pip."
        echo "        Dang nhap lam root va chay:"
        echo "            apt-get update && apt-get install -y python3-pip python3-venv"
        echo "        Hoac dung venv (khong can pip he thong):"
        echo "            python3 -m venv ~/venv-sft && source ~/venv-sft/bin/activate"
        exit 1
    fi
    echo "  -> pip OK: $("${PY}" -m pip --version | cut -d' ' -f1-2)"
fi

# Bao som: kiem tra GPU NVIDIA duoc nhin thay (WSL/Windows: can driver NVIDIA phia Windows,
# KHONG cai driver qua apt trong WSL). Nuit som de khong mat thoi gian tai torch 2.5 GB roi moi loi.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    echo "  -> GPU visible:"
    nvidia-smi -L | sed 's/^/       /'
else
    echo "[ERROR] Khong tim thay GPU NVIDIA (nvidia-smi khong chay duoc)."
    echo "        WSL: cai driver GeForce moi nhat tren Windows, chay 'wsl --update' trong PowerShell,"
    echo "             roi mo lai terminal. KHONG 'apt install' driver NVIDIA ben trong WSL."
    echo "        May Linux thu: kiem tra 'lspci | grep -i nvidia' va dau vao driver."
    exit 1
fi

# pip co the bi chan boi PEP 668 (Debian 12 / Ubuntu 24.04) -> thu --break-system-packages
pip_install() {
    if ! "${PY}" -m pip install -q "$@" 2>/dev/null; then
        "${PY}" -m pip install -q --break-system-packages "$@" 2>/dev/null \
            || "${PY}" -m pip install -q --user "$@"
    fi
}

echo "=============================================================================="
echo "  SFT CoT 81 Military Classes (Linux)"
echo "  MODEL_ID    = ${MODEL_ID}"
echo "  DATA        = ${LOCAL_DATA_DIR}"
echo "  OUTPUT_DIR  = ${OUTPUT_DIR}"
echo "  NUM_GPUS    = ${NUM_GPUS}"
echo "  BATCH_SIZE  = ${BATCH_SIZE} (GRAD_ACCUM = ${GRAD_ACCUM}, EPOCHS = ${EPOCHS}, LR = ${LR})"
echo "=============================================================================="

echo
echo "[1/3] Checking PyTorch >= 2.6 + CUDA and required packages..."
if ! "${PY}" -c "import sys, torch; v=tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]); sys.exit(0 if (v >= (2, 6) and torch.cuda.is_available()) else 1)" 2>/dev/null; then
    echo "  -> Installing PyTorch >= 2.6 (source: ${TORCH_SOURCE})..."
    case "${TORCH_SOURCE}" in
        CUDA_12_6) pip_install --upgrade "torch>=2.6" torchvision --index-url https://download.pytorch.org/whl/cu126 ;;
        CUDA_12_4) pip_install --upgrade "torch>=2.6" torchvision --index-url https://download.pytorch.org/whl/cu124 ;;
        PYPI)      pip_install --upgrade torch torchvision ;;
        *)         echo "[ERROR] TORCH_SOURCE khong hop le: ${TORCH_SOURCE} (dung CUDA_12_6 | CUDA_12_4 | PYPI)"; exit 1 ;;
    esac
fi
pip_install -U "transformers>=4.57" accelerate peft \
    qwen-vl-utils datasets pyarrow huggingface_hub pillow tqdm tensorboard
"${PY}" -c "import torch, transformers, peft; print('  -> Env OK: torch=' + torch.__version__ + ' | cuda=' + str(torch.cuda.is_available()) + ' | transformers=' + transformers.__version__ + ' | peft=' + peft.__version__)"

# cuda.is_available() va get_device_properties() van co the thanh cong khi wheel
# khong co kernel cho GPU (vi du cu126 tren Blackwell sm_120).
if ! "${PY}" - <<'PYCUDACHECK'
import sys
import torch

try:
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() = False")
    for device in range(torch.cuda.device_count()):
        with torch.cuda.device(device):
            x = torch.empty(16, device=f"cuda:{device}").uniform_(0.01, 16).log_()
            torch.cuda.synchronize(device)
            print(f"  -> CUDA compute OK: GPU {device}, {torch.cuda.get_device_name(device)}, "
                  f"sm_{''.join(map(str, torch.cuda.get_device_capability(device)))}, "
                  f"torch={torch.__version__}, CUDA={torch.version.cuda}")
except Exception as exc:
    print(f"[ERROR] PyTorch khong chay duoc kernel tren GPU: {exc}", file=sys.stderr)
    sys.exit(1)
PYCUDACHECK
then
    echo "[ERROR] Dung truoc khi tai model; checkpoint chua bi thay doi."
    echo "        Python cua script: $(command -v "${PY}")"
    echo "        Neu GPU la Blackwell sm_120 va torch dang la +cu126, cai wheel cu130:"
    echo "        ${PY} -m pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu130"
    echo "        Kiem tra driver NVIDIA ho tro CUDA 13.0 bang nvidia-smi; sau do chay lai script."
    exit 1
fi

echo
echo "[2/3] Checking Parquet dataset (${HF_DATASET_REPO})..."
export HF_DATASET_REPO LOCAL_DATA_DIR
"${PY}" - <<'PYCHECK'
import os
from pathlib import Path
from huggingface_hub import snapshot_download
from datasets import load_dataset

repo = os.environ["HF_DATASET_REPO"]
out = Path(os.environ["LOCAL_DATA_DIR"])
out.mkdir(parents=True, exist_ok=True)
pdir = out / "data"


def scan():
    if not pdir.is_dir():
        return [], []
    return (sorted(str(p) for p in pdir.glob("train-*.parquet")),
            sorted(str(p) for p in pdir.glob("validation-*.parquet")))


tf, vf = scan()
if len(tf) < 2 or len(vf) < 1:
    snapshot_download(repo_id=repo, repo_type="dataset",
                      allow_patterns=["data/*.parquet", "*.jsonl"],
                      local_dir=str(out))
    tf, vf = scan()
if not tf:
    raise SystemExit(f"Khong tim thay file .parquet nao trong {pdir}")

data_files = {"train": tf}
if vf:
    data_files["validation"] = vf
ds = load_dataset("parquet", data_files=data_files)
val_n = len(ds["validation"]) if "validation" in ds else 0
print(f"  -> Dataset OK: Train = {len(ds['train'])} | Val = {val_n} | sample image = {ds['train'][0]['image'].size}")
PYCHECK

# ----------------------------------------------------------------------------
# 3. Bao som: kiem tra VRAM du cho model 16-bit (Khong dung QLoRA 4-bit)
# ----------------------------------------------------------------------------
read -r VRAM_GB GPU_NAME <<EOF2
$("${PY}" -c "import torch
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f'{p.total_memory / 1024**3:.1f}', p.name.replace(' ', '_'))
else:
    print('0.0', 'no_cuda')")
EOF2
echo "  -> GPU: ${GPU_NAME} | VRAM: ${VRAM_GB} GB"
if awk "BEGIN { exit !(${VRAM_GB} < 20) }"; then
    echo "[ERROR] Model ${MODEL_ID} chay 16-bit (LoRA, khong quantize) can GPU >= 20 GB"
    echo "        (weights bf16 ~18 GB + activation/logits + optimizer)."
    echo "        May co ${VRAM_GB} GB: dung Kaggle Note L4 (24 GB) / A100 (40 GB),"
    echo "        Can GPU du VRAM de chay Qwen3.5-9B LoRA 16-bit, khong ho tro 4-bit."
    exit 1
fi
echo "  -> VRAM >= 20 GB: huan thuan cho LoRA 16-bit (BF16), khong quantize."

# ----------------------------------------------------------------------------
# 4. Kiem tra HF token neu bat day checkpoint len Hub
# ----------------------------------------------------------------------------
TRAIN_ARGS_EXTRA=()
if [ -n "${HF_PUSH_REPO}" ]; then
    if [ -z "${HF_TOKEN:-}" ] && [ ! -f "${HOME}/.cache/huggingface/token" ] && [ ! -f "${HOME}/.hf_token" ]; then
        echo "[WARN] Chua thay HF_TOKEN. Day checkpoint len '${HF_PUSH_REPO}' co the se that bai."
        echo "       Hay chay:  export HF_TOKEN=hf_xxxxxxxx   (hoac: huggingface-cli login)"
    fi
    TRAIN_ARGS_EXTRA+=(--hub_repo "${HF_PUSH_REPO}")
    if [ "${HF_REPO_PUBLIC}" = "1" ]; then
        TRAIN_ARGS_EXTRA+=(--hub_public)
    fi
    echo "  -> Se tu day checkpoint + final_adapter len: ${HF_PUSH_REPO} (private=${HF_REPO_PUBLIC})"
else
    echo "  -> HF_PUSH_REPO trong -> khong day len Hub."
fi

# Bao som neu model vuot qua VRAM (tranh mat hang gio tai xong crash)
"${PY}" - "${MODEL_ID}" "${VRAM_GB}" <<'PYPREFLIGHT'
import sys
from huggingface_hub import HfApi

model_id, vram_gb = sys.argv[1], float(sys.argv[2])
try:
    info = HfApi().model_info(model_id, files_metadata=True)
    gb = sum(f.size or 0 for f in info.siblings
             if str(f.rfilename).endswith(".safetensors")) / 1024**3
except Exception as exc:
    print(f"  -> [preflight] Bo qua kiem tra kich thuoc ({type(exc).__name__})")
    sys.exit(0)
# LoRA 16-bit: toan bo weights (bf16) + ~3 GB activation/logits/optimizer (gradient checkpointing, bs nho)
need = gb + 3.0
print(f"  -> [preflight] {model_id}: {gb:.1f} GB weights (bf16) | uoc tinh can ~{need:.1f} GB tren {vram_gb:.1f} GB VRAM")
if need > vram_gb:
    print(f"[ERROR] GPU KHONG du cho model nay ({need:.1f} GB > {vram_gb:.1f} GB).")
    print("        Chay tren may GPU >= 24 GB (L4/A100/4090) hoac Kaggle Note.")
    print("        Can GPU du VRAM de chay model 16-bit, khong ho tro 4-bit.")
    sys.exit(1)
PYPREFLIGHT

echo
echo "[3/3] Starting SFT Training (enable_thinking=True, LoRA 16-bit) on ${MODEL_ID}..."
TB_LOG_DIR="${OUTPUT_DIR}/runs"
mkdir -p "${TB_LOG_DIR}"
if [ "${START_TB}" = "1" ]; then
    pkill -f "tensorboard.*--logdir.*${TB_LOG_DIR}" >/dev/null 2>&1 || true
    nohup "${PY}" -m tensorboard.main --logdir "${TB_LOG_DIR}" --host 0.0.0.0 --port "${TB_PORT}" > "${OUTPUT_DIR}/tensorboard.log" 2>&1 &
    echo "  -> TensorBoard dang chay ngam tai port ${TB_PORT} (logdir: ${TB_LOG_DIR})"
    echo "     Xem tren may ca nhan qua SSH tunnel:  ssh -N -L ${TB_PORT}:localhost:${TB_PORT} <user>@<ip-may-ao> -p <ssh-port>"
    echo "     Sau do mo trinh duyet:                http://localhost:${TB_PORT}"
fi

TRAIN_ARGS=(
    --model_id "${MODEL_ID}"
    --parquet_dir "${LOCAL_DATA_DIR}"
    --output_dir "${OUTPUT_DIR}"
    --epochs "${EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --grad_accum "${GRAD_ACCUM}"
    --lr "${LR}"
    --lora_r "${LORA_R}"
    --lora_alpha "${LORA_ALPHA}"
    --save_steps "${SAVE_STEPS}"
    --max_seq_length "${MAX_SEQ_LEN}"
)
if [ "${#TRAIN_ARGS_EXTRA[@]}" -gt 0 ]; then
    TRAIN_ARGS+=("${TRAIN_ARGS_EXTRA[@]}")
fi

if [ "${NUM_GPUS}" -gt 1 ]; then
    torchrun --nproc_per_node="${NUM_GPUS}" "${SCRIPT_DIR}/train_sft_qwen4b_81classes.py" "${TRAIN_ARGS[@]}"
else
    "${PY}" "${SCRIPT_DIR}/train_sft_qwen4b_81classes.py" "${TRAIN_ARGS[@]}"
fi

echo
echo "=============================================================================="
echo "  COMPLETED. Final LoRA adapter saved at:"
echo "  ${OUTPUT_DIR}/final_adapter"
if [ -n "${HF_PUSH_REPO}" ]; then
    echo "  Da day len HuggingFace:  https://huggingface.co/${HF_PUSH_REPO}/tree/main/final_adapter"
fi
echo "=============================================================================="
