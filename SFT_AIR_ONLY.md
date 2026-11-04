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

## Sinh lại prompt và CoT Air-only bằng API

File `rewrite_air_cot_api.py` đọc ảnh thật từ Parquet và gọi vision API để
rewrite `assistant_cot`. Đây là **minimal rewrite**: giữ nguyên văn phong,
4 mục reasoning và format của prompt 81 lớp; chỉ đổi inventory sang 30 class
Air-only, bỏ Grd/Sea khỏi phần phân loại và loại các tham chiếu non-Air. File
gốc không bị sửa; output là dataset JSONL mới. Script tự resume theo `id` nếu
bị ngắt.

Ví dụ dùng OpenAI API, chạy thử 5 dòng mỗi split trước:

```powershell
cd D:\du_lieu_that\up_github_1
$env:OPENAI_API_KEY = "sk-..."
python rewrite_air_cot_api.py `
  --dataset-dir D:\du_lieu_that\hf_sft_air_only_parquet `
  --output-dir D:\du_lieu_that\hf_sft_air_only_v2 `
  --model gpt-4o-mini `
  --limit 5 `
  --workers 4
```

Chạy toàn bộ sau khi kiểm tra thử:

```powershell
python rewrite_air_cot_api.py `
  --dataset-dir D:\du_lieu_that\hf_sft_air_only_parquet `
  --output-dir D:\du_lieu_that\hf_sft_air_only_v2 `
  --model gpt-4o-mini `
  --workers 4
```

Output:

```text
D:\du_lieu_that\hf_sft_air_only_v2\train-air-only-v2.jsonl
D:\du_lieu_that\hf_sft_air_only_v2\validation-air-only-v2.jsonl
```

Script bắt buộc output phải có `<think>...</think>`, `<answer>Air.X</answer>`,
answer phải khớp `gt_class`, và không được còn `Grd.`, `Sea.`, `81 classes`,
`ground vehicle` hoặc `naval vessel`. Dòng API lỗi/không hợp lệ sẽ retry và
không được ghi vào file.

Có thể dùng endpoint OpenAI-compatible khác bằng `--base-url`, ví dụ:

```powershell
python rewrite_air_cot_api.py ... --base-url https://openrouter.ai/api/v1 --model openai/gpt-4o-mini
```