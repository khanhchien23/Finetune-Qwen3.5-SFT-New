# SFT riêng cho 30 lớp máy bay

Dataset đã lọc tại:

```text
D:\du_lieu_that\hf_sft_air_only_parquet
```

Gồm `4,038` mẫu train và `449` mẫu validation, chỉ có `gt_class` bắt đầu bằng
`Air.`. Dataset gốc không bị sửa. Script SFT hiện tại đọc trực tiếp Parquet và
giữ nguyên ảnh, system prompt, user prompt và assistant CoT.

Chạy local/Linux/Modal container:

```bash
cd /path/to/up_github_1
chmod +x run_sft_air_only.sh
 HF_TOKEN=hf_xxx \\
HUB_REPO=KhanhChien/FineTuneQwen35-9B-Air-Only \\
./run_sft_air_only.sh
```

Nếu muốn thử ngắn trước:

```bash
EPOCHS=1 SAVE_STEPS=50 ./run_sft_air_only.sh
```

Trên H100 80GB, cấu hình mặc định là `batch_size=8` và `grad_accum=4`:
effective batch = 32, tương đương `batch_size=4, grad_accum=8` trên GPU 32GB.
Script vẫn giữ gradient checkpointing để an toàn VRAM ở sequence length 1280.

Script dùng **base Qwen/Qwen3.5-9B + LoRA SFT mới**, không nạp GRPO-304.
Đầu ra mặc định là `Qwen3.5-9B-SFT-Air-Only/final_adapter`. Sau khi SFT xong,
dùng chính adapter này để infer trên các folder `Air.*`.

## Xuất CoT gốc của dataset để kiểm tra bằng mắt

File `export_sft_cot_jsonl.py` xuất chính target đã dùng để train, không phải
CoT do model sinh ra:

```powershell
cd D:\du_lieu_that\up_github_1
python export_sft_cot_jsonl.py `
  --dataset-dir D:\du_lieu_that\hf_sft_air_only_parquet
```

Kết quả:

```text
D:\du_lieu_that\hf_sft_air_only_parquet\jsonl\train-air-only.jsonl
D:\du_lieu_that\hf_sft_air_only_parquet\jsonl\validation-air-only.jsonl
```

Mỗi dòng có `gt_class`, `system_prompt`, `user_prompt`, `assistant_cot` và
`messages`. Trường `assistant_cot` là toàn bộ chuỗi `<think>...</think>` và
`<answer>...</answer>` gốc. Mặc định ảnh chỉ ghi metadata để file dễ mở bằng
mắt. Có thể thử 5 dòng mỗi split bằng `--limit 5`.