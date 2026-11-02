#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Script huấn luyện GDPO (Group Decoupled Policy Optimization) cho Qwen3.5-9B (VLM)
khởi tạo từ SFT LoRA `checkpoint-2004` (`KhanhChien/FineTuneQwen35-9B-New`).

Sự khác biệt cốt lõi giữa GDPO và GRPO (chỉ dùng 2 Reward: Exact Accuracy + Format):
- Trong GRPO truyền thống:
    Cộng gộp r_i = w_acc * R_acc + w_fmt * R_fmt trước rồi mới chia cho std(r).
    Điều này làm tín hiệu Format có thể pha loãng hoặc làm méo biên độ của tín hiệu Accuracy.
- Trong GDPO (Group Decoupled Policy Optimization):
    1. Tách rời (Decouple) chuẩn hóa từng Reward riêng biệt trong nhóm G câu trả lời:
         A_acc_i = (R_acc_i - mean(R_acc)) / (std(R_acc) + 1e-4)   (nếu std(R_acc) > 0, ngược lại = 0)
         A_fmt_i = (R_fmt_i - mean(R_fmt)) / (std(R_fmt) + 1e-4)   (nếu std(R_fmt) > 0, ngược lại = 0)
    2. Tổng hợp Advantage đã tách rời và chuẩn hóa Whitening:
         A_raw_i  = w_acc * A_acc_i + w_fmt * A_fmt_i
         A_gdpo_i = (A_raw_i - mean(A_raw)) / (std(A_raw) + 1e-4)
    Nhờ vậy, khi cả G câu trả lời đều đúng format (std(R_fmt) == 0 -> A_fmt_i = 0),
    tín hiệu Accuracy được giữ nguyên 100% độ sắc nét; khi có câu sai format, A_fmt_i
    phạt độc lập câu đó mà không làm lệch chuẩn của các câu trả lời đúng nhãn!
"""

import os
import sys
import re
import json
import math
import time
import random
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.modules["torchao"] = None

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText, get_cosine_schedule_with_warmup
import peft.import_utils
peft.import_utils.is_torchao_available = lambda: False
import peft.tuners.lora.torchao
peft.tuners.lora.torchao.is_torchao_available = lambda: False
from peft import PeftModel

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ==============================================================================
# 1. DANH SÁCH 81 LỚP CHUẨN & SYSTEM/USER PROMPT (KHỚP 100% VỚI SFT & INFERENCE)
# ==============================================================================
MIN_PIXELS = 64 * 28 * 28
MAX_PIXELS = 256 * 28 * 28

VALID_81_CLASSES = (
    "Air.A10", "Air.AH64", "Air.AS365", "Air.B2", "Air.B29", "Air.C130", "Air.C17",
    "Air.Casa212", "Air.CH47", "Air.CH53", "Air.F22", "Air.F35", "Air.IL76", "Air.J7",
    "Air.JH7", "Air.Ka27", "Air.Ka52", "Air.KC135", "Air.L39", "Air.Mi24", "Air.Mi28",
    "Air.Mi8", "Air.Mig29", "Air.Rafale", "Air.Su22", "Air.Su27", "Air.T6Texan",
    "Air.Yak130", "Air.Yak52", "Air.Z8", "Grd.2S1", "Grd.BM21", "Grd.BM27",
    "Grd.BM30", "Grd.BMP2", "Grd.Bradley", "Grd.BRDM2", "Grd.BTR152", "Grd.BTR60",
    "Grd.Gaz66", "Grd.Himars", "Grd.Humvee", "Grd.Kamaz", "Grd.Kraz255", "Grd.LAV25",
    "Grd.M113", "Grd.M1Abrams", "Grd.Maz537", "Grd.Merkava", "Grd.MTLB", "Grd.PT76",
    "Grd.SU100", "Grd.SU76", "Grd.T34", "Grd.T55", "Grd.T90", "Grd.Ural4320",
    "Grd.Zil131", "Grd.ZSU234", "Grd.ZSU57", "Sea.ArleighBurke", "Sea.Buyan",
    "Sea.GeraldFord", "Sea.Gorshkov", "Sea.Independence", "Sea.Kilo", "Sea.Kirov",
    "Sea.Molniya", "Sea.Nanuchka", "Sea.OsaII", "Sea.Ticonderoga",
    "Sea.Type002.Shandong", "Sea.Type022.Houbei", "Sea.Type051B.Luhai",
    "Sea.Type053H3.JiangweiII", "Sea.Type054A.JiangkaiII", "Sea.Type055.Renhai",
    "Sea.TypeAsahi", "Sea.TypeChengKung", "Sea.TypeKangDing", "Sea.Wasp",
)

SYSTEM_PROMPT = f"""You are an expert military analyst specializing in Fine-Grained Military Vehicle Recognition.
Your task is to inspect the provided image and classify the military vehicle into EXACTLY ONE of the following 81 target classes:

{", ".join(VALID_81_CLASSES)}

Follow this structured reasoning process before making your final decision:
1. Viewpoint & Environment: Identify the camera viewing angle and operational setting.
2. Coarse Category: Determine the primary domain (Air / Grd / Sea) and general vehicle type.
3. Fine-Grained Visual Features: Describe the distinctive physical characteristics strictly visible in the image.
4. Disambiguation & Conclusion: Rule out visually similar candidate types based on observed features and select the exact target class name.

Required output format:
<think>
Write your step-by-step visual analysis here...
</think>
<answer>Exact_Class_Name</answer>"""

USER_PROMPT = (
    "Inspect this military vehicle carefully. Provide a structured 4-step visual analysis "
    "inside <think>...</think> and conclude with the exact class label inside <answer>...</answer>."
)


# ==============================================================================
# 2. HÀM TÍNH 2 REWARD RIÊNG BIỆT (ACCURACY + FORMAT) & DECOUPLED ADVANTAGE (GDPO)
# ==============================================================================
def extract_answer(raw_text: str) -> str:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", raw_text, re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_think(raw_text: str) -> str:
    match = re.search(r"<think>\s*(.*?)\s*</think>", raw_text, re.DOTALL)
    return match.group(1).strip() if match else ""


def compute_gdpo_rewards(raw_completion: str, gt_class: str) -> Tuple[float, float]:
    """
    Trả về cặp 2 Reward hoàn toàn tách biệt (r_acc, r_format) thuộc {0.0, 1.0}:
      1. r_acc    (0.0 hoặc 1.0): Nhãn trong <answer>...</answer> khớp chính xác `gt_class`.
      2. r_format (0.0 hoặc 1.0): Đóng thẻ </think> trước khi mở <answer>Exact_Class</answer>
                                  và nội dung <think> không rỗng.
    """
    pred = extract_answer(raw_completion)
    think_body = extract_think(raw_completion)

    has_closed_think = "</think>" in raw_completion and len(think_body) > 10
    has_closed_answer = bool(re.search(r"<answer>\s*[^\s<>]+\s*</answer>", raw_completion))
    valid_order = False
    if has_closed_think and has_closed_answer:
        valid_order = raw_completion.find("</think>") < raw_completion.find("<answer>")
    r_format = 1.0 if valid_order else 0.0

    pred_cf = pred.strip().casefold()
    gt_cf = gt_class.strip().casefold()
    r_acc = 1.0 if (pred_cf and pred_cf == gt_cf) else 0.0

    return r_acc, r_format


def compute_gdpo_advantages(
    r_acc_list: List[float],
    r_fmt_list: List[float],
    w_acc: float = 1.0,
    w_fmt: float = 0.2,
    device: torch.device = torch.device("cpu"),
) -> Optional[torch.Tensor]:
    """
    Thuật toán GDPO (Group Decoupled Policy Optimization):
      1. Chuẩn hóa riêng r_acc trong nhóm G:
           A_acc = (r_acc - mean(r_acc)) / (std(r_acc) + 1e-4)  nếu std(r_acc) > 1e-4, ngược lại = 0
      2. Chuẩn hóa riêng r_fmt trong nhóm G:
           A_fmt = (r_fmt - mean(r_fmt)) / (std(r_fmt) + 1e-4)  nếu std(r_fmt) > 1e-4, ngược lại = 0
      3. Nếu cả 2 nhánh đều có std == 0 -> trả về None (bỏ qua nhóm đồng nhất).
      4. Kết hợp có trọng số và chuẩn hóa Whitening lần cuối:
           A_raw  = w_acc * A_acc + w_fmt * A_fmt
           A_gdpo = (A_raw - mean(A_raw)) / (std(A_raw) + 1e-4)
    """
    acc_t = torch.tensor(r_acc_list, dtype=torch.float32, device=device)
    fmt_t = torch.tensor(r_fmt_list, dtype=torch.float32, device=device)

    std_acc = acc_t.std()
    std_fmt = fmt_t.std()

    if std_acc.item() < 1e-4 and std_fmt.item() < 1e-4:
        return None

    if std_acc.item() >= 1e-4:
        a_acc = (acc_t - acc_t.mean()) / (std_acc + 1e-4)
    else:
        a_acc = torch.zeros_like(acc_t)

    if std_fmt.item() >= 1e-4:
        a_fmt = (fmt_t - fmt_t.mean()) / (std_fmt + 1e-4)
    else:
        a_fmt = torch.zeros_like(fmt_t)

    a_raw = w_acc * a_acc + w_fmt * a_fmt
    std_raw = a_raw.std()
    if std_raw.item() < 1e-4:
        return None

    a_gdpo = (a_raw - a_raw.mean()) / (std_raw + 1e-4)
    return a_gdpo


# ==============================================================================
# 3. TẢI DỮ LIỆU PARQUET TỪ HUGGINGFACE VÀ TẢI ADAPTER CHECKPOINT-2004
# ==============================================================================
def load_gdpo_dataset(dataset_repo_or_dir: str, max_samples: int = 0, seed: int = 42) -> List[Dict[str, Any]]:
    from datasets import load_dataset
    p = Path(dataset_repo_or_dir)
    if p.is_dir():
        parquet_files = sorted(p.rglob("*.parquet"))
        train_files = [str(f) for f in parquet_files if "train" in f.name.lower()] or [str(f) for f in parquet_files]
        ds = load_dataset("parquet", data_files={"train": train_files})["train"]
    else:
        ds = load_dataset(dataset_repo_or_dir)["train"]

    indices = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    if max_samples > 0:
        indices = indices[:max_samples]
    return [ds[i] for i in indices]


def extract_item_image_and_gt(item: Dict[str, Any]) -> Tuple[Image.Image, str]:
    raw_img = item.get("image")
    if isinstance(raw_img, Image.Image):
        img = raw_img.convert("RGB")
    elif isinstance(raw_img, dict) and raw_img.get("bytes"):
        import io
        img = Image.open(io.BytesIO(raw_img["bytes"])).convert("RGB")
    else:
        raise ValueError("Mẫu dữ liệu không chứa ảnh hợp lệ trong cột `image`.")

    gt_class = item.get("gt_class", "")
    if not gt_class:
        assistant_cot = item.get("assistant_cot", "")
        if not assistant_cot and "messages" in item:
            assistant_cot = item["messages"][-1]["content"]
        gt_class = extract_answer(assistant_cot)
    if not gt_class:
        raise ValueError("Không trích xuất được nhãn gt_class từ mẫu dữ liệu.")
    return img, gt_class.strip()

def resolve_sft_checkpoint(
    sft_repo_or_dir: str, ckpt_name: str = "checkpoint-2004", cache_dir: str = "./sft_ckpt_cache"
) -> Path:
    local_cand = (
        Path(sft_repo_or_dir) / ckpt_name
        if (Path(sft_repo_or_dir) / ckpt_name).is_dir()
        else Path(sft_repo_or_dir)
    )
    if (local_cand / "adapter_config.json").is_file() and (local_cand / "adapter_model.safetensors").is_file():
        if (local_cand / "adapter_model.safetensors").stat().st_size > 1_000_000:
            print(f"--> Tìm thấy SFT adapter hợp lệ tại local: {local_cand}")
            return local_cand

    from huggingface_hub import hf_hub_download
    target_root = Path(cache_dir)
    resolved = target_root / ckpt_name
    resolved.mkdir(parents=True, exist_ok=True)
    print(f"--> Đang tải SFT `{ckpt_name}` từ HuggingFace ({sft_repo_or_dir})...")
    for fname in ("adapter_config.json", "adapter_model.safetensors"):
        hf_hub_download(
            repo_id=sft_repo_or_dir,
            filename=f"{ckpt_name}/{fname}",
            local_dir=str(target_root),
            force_download=True,
        )
    weights = resolved / "adapter_model.safetensors"
    if not weights.is_file() or weights.stat().st_size < 1_000_000:
        raise RuntimeError(f"Adapter tải về tại {resolved} bị lỗi hoặc là LFS pointer!")
    print(f"--> Đã tải xong `{ckpt_name}` ({weights.stat().st_size / 1024**2:.1f} MB) tại: {resolved}")
    return resolved


# ==============================================================================
# 4. HÀM TÍNH LOG-PROBS TRÊN CÁC TOKEN SINH RA & ĐẨY CHECKPOINT LÊN HUB
# ==============================================================================
def compute_completion_logprobs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    prompt_len: int,
    prompt_mm_token_type_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    G, total_len = input_ids.shape
    comp_len = total_len - prompt_len
    pv_expanded = pixel_values.repeat(G, *([1] * (pixel_values.dim() - 1)))
    thw_expanded = image_grid_thw.repeat(G, 1)

    extra_kwargs: Dict[str, Any] = {}
    if prompt_mm_token_type_ids is not None:
        mm_zeros = torch.zeros(
            (G, comp_len),
            dtype=prompt_mm_token_type_ids.dtype,
            device=prompt_mm_token_type_ids.device,
        )
        extra_kwargs["mm_token_type_ids"] = torch.cat(
            [prompt_mm_token_type_ids.repeat(G, 1), mm_zeros], dim=1
        )

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pv_expanded,
        image_grid_thw=thw_expanded,
        use_cache=False,
        **extra_kwargs,
    )
    completion_logits = outputs.logits[:, prompt_len - 1 : -1, :]  # [G, C, V]
    completion_ids = input_ids[:, prompt_len:]                     # [G, C]
    completion_mask = attention_mask[:, prompt_len:].float()       # [G, C]

    # Với G=4 trên GPU 40GB-48GB VRAM, tính log_softmax vector hóa toàn batch [G, C, V] để tăng tốc tối đa
    log_probs = F.log_softmax(completion_logits.float(), dim=-1)
    per_token_logps = torch.gather(log_probs, dim=-1, index=completion_ids.unsqueeze(-1)).squeeze(-1)  # [G, C]
    return per_token_logps, completion_mask


def push_gdpo_checkpoint_to_hub(folder: Path, repo_id: str, path_in_repo: str, runs_dir: Optional[Path] = None):
    try:
        from huggingface_hub import HfApi, create_repo
        create_repo(repo_id=repo_id, private=True, exist_ok=True)
        api = HfApi()
        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(folder),
            path_in_repo=path_in_repo,
            commit_message=f"Upload GDPO {path_in_repo}",
        )
        print(f"--> [Hub] Đã đẩy {path_in_repo} lên HuggingFace: {repo_id}/{path_in_repo}", flush=True)
        if runs_dir and runs_dir.is_dir():
            api.upload_folder(
                repo_id=repo_id,
                folder_path=str(runs_dir),
                path_in_repo="runs",
                commit_message=f"Update GDPO TensorBoard logs ({path_in_repo})",
            )
    except Exception as exc:
        print(f"--> [Hub WARN] Không đẩy được lên {repo_id}: {type(exc).__name__}: {exc}", flush=True)

# ==============================================================================
# 5. VÒNG LẶP HUẤN LUYỆN GDPO (GROUP DECOUPLED POLICY OPTIMIZATION) CHÍNH
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train GDPO on Qwen3.5-9B starting from SFT checkpoint-2004")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--sft_repo", type=str, default="KhanhChien/FineTuneQwen35-9B-New")
    parser.add_argument("--sft_checkpoint", type=str, default="checkpoint-2004")
    parser.add_argument("--dataset", type=str, default="KhanhChien/Data-Qwen3.5-SFT-New")
    parser.add_argument("--output_dir", type=str, default="./Qwen3.5-9B-GDPO-ckpt2004")
    parser.add_argument("--hub_repo", type=str, default="KhanhChien/GDPO-Qwen35-9B-New")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Số epoch duyệt qua toàn bộ tập dữ liệu (mặc định: 1 epoch = toàn bộ dataset)")
    parser.add_argument("--max_steps", type=int, default=0,
                        help="0 = Không giới hạn, chạy hết toàn bộ bộ dữ liệu; >0 nếu muốn dừng sớm theo step")
    parser.add_argument("--num_generations", type=int, default=4,
                        help="Số lượng câu trả lời G sinh ra cho mỗi ảnh (mặc định G=4)")
    parser.add_argument("--grad_accum", type=int, default=8,
                        help="Số ảnh có phương sai > 0 tích lũy gradient cho 1 bước cập nhật")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--kl_beta", type=float, default=0.02)
    parser.add_argument("--w_acc", type=float, default=1.0, help="Trọng số nhánh Accuracy sau khi chuẩn hóa tách rời")
    parser.add_argument("--w_fmt", type=float, default=0.2, help="Trọng số nhánh Format sau khi chuẩn hóa tách rời")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max_new_tokens", type=int, default=384)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="Qwen35-9B-RL-81Classes",
                        help="Tên project trên WandB để vẽ chung biểu đồ so sánh GRPO vs GDPO")
    parser.add_argument("--wandb_run_name", type=str, default="GDPO-ckpt2004",
                        help="Tên run hiển thị trên WandB")
    parser.add_argument("--no_wandb", action="store_true", help="Tắt logging lên WandB")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("Cần GPU CUDA (khuyên dùng A100 40GB/80GB hoặc H100 trên Modal) để train GDPO!")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "runs" / args.wandb_run_name
    runs_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(runs_dir)) if HAS_TB else None

    use_wandb = HAS_WANDB and (not args.no_wandb) and bool(os.environ.get("WANDB_API_KEY"))
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )
        print(f"--> [WandB] Đã bật theo dõi trực tiếp tại project: {args.wandb_project} ({args.wandb_run_name})", flush=True)
    else:
        print(f"--> [TensorBoard] Đang ghi log tại: {runs_dir} (Đặt biến môi trường WANDB_API_KEY nếu muốn bật thêm WandB)", flush=True)

    sft_adapter_dir = resolve_sft_checkpoint(
        args.sft_repo, args.sft_checkpoint, cache_dir=str(out_dir / "sft_base_cache")
    )

    print(f"--> Đang tải Processor & Base Model: {args.model_id} ({dtype})...", flush=True)
    try:
        processor = AutoProcessor.from_pretrained(
            args.model_id,
            size={"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS},
            trust_remote_code=True,
        )
    except Exception:
        processor = AutoProcessor.from_pretrained(
            args.model_id,
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS,
            trust_remote_code=True,
        )
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "right"

    base_model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        dtype=dtype,
        device_map={"": 0},
        trust_remote_code=True,
    )
    base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base_model.enable_input_require_grads()

    model = PeftModel.from_pretrained(
        base_model, str(sft_adapter_dir), adapter_name="policy", is_trainable=True
    )
    model.load_adapter(str(sft_adapter_dir), adapter_name="ref", is_trainable=False)
    model.set_adapter("policy")
    model.print_trainable_parameters()

    processor.save_pretrained(str(out_dir))

    # 3. Tải TOÀN BỘ dữ liệu huấn luyện (max_samples=0 -> lấy hết 100% dataset)
    dataset = load_gdpo_dataset(args.dataset, max_samples=0, seed=args.seed)
    total_samples = len(dataset) * max(1, args.epochs)
    est_total_steps = args.max_steps if args.max_steps > 0 else max(1, total_samples // args.grad_accum)
    print(
        f"--> Đã nạp toàn bộ {len(dataset):,} mẫu dữ liệu (x {args.epochs} epoch = {total_samples:,} mẫu) | "
        f"G={args.num_generations}, grad_accum={args.grad_accum} (tối ưu ~38GB/48GB VRAM, chỉ train LoRA).",
        flush=True,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=max(10, int(est_total_steps * 0.05)), num_training_steps=est_total_steps
    )

    pad_token_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    eos_token_id = processor.tokenizer.eos_token_id

    global_step = 0
    sample_idx = 0
    optimizer.zero_grad()

    pbar = tqdm(total=total_samples, desc="GDPO Dataset Progress", unit="img")
    while sample_idx < total_samples and (args.max_steps <= 0 or global_step < args.max_steps):
        step_loss_sum = 0.0
        step_kl_sum = 0.0
        step_acc_sum = 0.0
        step_fmt_sum = 0.0
        valid_groups = 0
        attempts_in_step = 0

        while valid_groups < args.grad_accum and sample_idx < total_samples:
            item = dataset[sample_idx % len(dataset)]
            sample_idx += 1
            attempts_in_step += 1
            pbar.update(1)

            image, gt_class = extract_item_image_and_gt(item)
            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": USER_PROMPT},
                ]},
            ]
            prompt_text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
            )
            prompt_inputs = processor(
                text=[prompt_text], images=[image], return_tensors="pt"
            ).to(device)
            prompt_len = prompt_inputs["input_ids"].size(1)

            # --- BƯỚC A: ROLLOUT G CÂU TRẢ LỜI TỪ POLICY HIỆN TẠI ---
            model.eval()
            model.set_adapter("policy")
            with torch.no_grad():
                gen_ids = model.generate(
                    **prompt_inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=0.95,
                    num_return_sequences=args.num_generations,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                ).clone()

            comp_ids = gen_ids[:, prompt_len:]
            decoded_list = processor.batch_decode(comp_ids, skip_special_tokens=False)
            r_acc_list = []
            r_fmt_list = []
            for text in decoded_list:
                full_completion = "<think>\n" + text
                r_acc, r_fmt = compute_gdpo_rewards(full_completion, gt_class)
                r_acc_list.append(r_acc)
                r_fmt_list.append(r_fmt)

            step_acc_sum += sum(r_acc_list) / len(r_acc_list)
            step_fmt_sum += sum(r_fmt_list) / len(r_fmt_list)

            # Tính Decoupled Advantage theo đúng công thức GDPO
            advantages = compute_gdpo_advantages(
                r_acc_list, r_fmt_list, w_acc=args.w_acc, w_fmt=args.w_fmt, device=device
            )
            if advantages is None:
                continue

            G, total_len = gen_ids.shape
            attn_mask = torch.ones_like(gen_ids, dtype=torch.long)
            for i in range(G):
                eos_positions = (gen_ids[i, prompt_len:] == eos_token_id).nonzero(as_tuple=True)[0]
                if len(eos_positions) > 0:
                    first_eos = prompt_len + int(eos_positions[0].item())
                    if first_eos + 1 < total_len:
                        attn_mask[i, first_eos + 1 :] = 0

            # --- BƯỚC B: TÍNH LOGP CỦA REFERENCE MODEL (checkpoint-2004 gốc) ---
            prompt_mm_ids = prompt_inputs.get("mm_token_type_ids")
            with torch.no_grad():
                model.set_adapter("ref")
                ref_logps, comp_mask = compute_completion_logprobs(
                    model,
                    gen_ids,
                    attn_mask,
                    prompt_inputs["pixel_values"],
                    prompt_inputs["image_grid_thw"],
                    prompt_len,
                    prompt_mm_token_type_ids=prompt_mm_ids,
                )

            # --- BƯỚC C: TÍNH LOGP CỦA POLICY MODEL & GDPO LOSS ---
            model.train()
            model.set_adapter("policy")
            policy_logps, comp_mask = compute_completion_logprobs(
                model,
                gen_ids,
                attn_mask,
                prompt_inputs["pixel_values"],
                prompt_inputs["image_grid_thw"],
                prompt_len,
                prompt_mm_token_type_ids=prompt_mm_ids,
            )

            log_ratio = ref_logps.detach() - policy_logps
            per_token_kl = torch.exp(log_ratio) - log_ratio - 1.0  # [G, C]

            adv_expanded = advantages.unsqueeze(1)  # [G, 1]
            per_token_loss = -(adv_expanded * policy_logps) + args.kl_beta * per_token_kl
            masked_loss = (per_token_loss * comp_mask).sum(dim=1) / comp_mask.sum(dim=1).clamp(min=1.0)
            loss = masked_loss.mean() / max(1, args.grad_accum)

            loss.backward()

            mean_kl = ((per_token_kl.detach() * comp_mask).sum() / comp_mask.sum().clamp(min=1.0)).item()
            step_loss_sum += loss.item() * args.grad_accum
            step_kl_sum += mean_kl
            valid_groups += 1

        if valid_groups == 0:
            continue

        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        global_step += 1

        avg_loss = step_loss_sum / max(1, valid_groups)
        avg_kl = step_kl_sum / max(1, valid_groups)
        avg_acc = step_acc_sum / max(1, attempts_in_step)
        avg_fmt = step_fmt_sum / max(1, attempts_in_step)
        avg_reward = args.w_acc * avg_acc + args.w_fmt * avg_fmt
        skipped_groups = attempts_in_step - valid_groups
        cur_lr = scheduler.get_last_lr()[0]

        print(
            f"[GDPO Step {global_step} | Img {sample_idx}/{total_samples}] "
            f"reward={avg_reward:.3f} | rollout_acc={avg_acc:.2%} | format_ok={avg_fmt:.2%} | "
            f"loss={avg_loss:.4f} | kl={avg_kl:.4f} | lr={cur_lr:.2e}",
            flush=True,
        )

        metrics = {
            "reward/total_mean": avg_reward,
            "reward/accuracy_rate": avg_acc,
            "reward/format_ok_rate": avg_fmt,
            "train/loss": avg_loss,
            "train/kl_divergence": avg_kl,
            "train/learning_rate": cur_lr,
            "train/skipped_easy_groups": skipped_groups,
            "train/samples_processed": sample_idx,
        }
        if tb_writer is not None:
            for k, v in metrics.items():
                tb_writer.add_scalar(k, v, global_step)
            tb_writer.flush()
        if use_wandb:
            wandb.log(metrics, step=global_step)

        is_finished = (sample_idx >= total_samples) or (args.max_steps > 0 and global_step >= args.max_steps)
        if global_step % args.save_steps == 0 or is_finished:
            ckpt_name = f"checkpoint-gdpo-{global_step}"
            ckpt_dir = out_dir / ckpt_name
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.set_adapter("policy")
            model.save_pretrained(str(ckpt_dir), selected_adapters=["policy"])
            sub_policy = ckpt_dir / "policy"
            if sub_policy.is_dir():
                for f in sub_policy.iterdir():
                    target = ckpt_dir / f.name
                    if target.exists():
                        target.unlink()
                    f.rename(target)
                sub_policy.rmdir()
            processor.save_pretrained(str(ckpt_dir))
            print(f"--> Đã lưu GDPO LoRA checkpoint tại: {ckpt_dir}", flush=True)
            if args.hub_repo:
                push_gdpo_checkpoint_to_hub(ckpt_dir, args.hub_repo, ckpt_name, runs_dir=runs_dir)

    pbar.close()
    if tb_writer is not None:
        tb_writer.close()
    if use_wandb:
        wandb.finish()
    print(f"=== HOÀN TẤT HUẤN LUYỆN GDPO TOÀN BỘ DATASET ({sample_idx:,} ẢNH, {global_step} STEPS) ===", flush=True)


if __name__ == "__main__":
    main()


