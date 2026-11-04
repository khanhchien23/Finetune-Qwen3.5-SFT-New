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
  --batch-size 16 `
  --limit 5 `
  --workers 4
```

Chạy toàn bộ sau khi kiểm tra thử:

```powershell
python rewrite_air_cot_api.py `
  --dataset-dir D:\du_lieu_that\hf_sft_air_only_parquet `
  --output-dir D:\du_lieu_that\hf_sft_air_only_v2 `
  --model gpt-4o-mini `
  --batch-size 16 `
  --workers 4
```

`--batch-size 16` nghĩa là mỗi wave submit tối đa 16 ảnh lên API. Endpoint
vision hiện tại vẫn gọi một request cho mỗi ảnh; `--workers 16` cho phép tối đa
16 request chạy đồng thời. Nếu proxy giới hạn rate, giảm xuống `--workers 4`
hoặc `--workers 8`.

Pipeline hiện encode ảnh JPEG nhẹ, không ép `detail=high`, giới hạn output 500
token và mặc định chạy 16 request đồng thời để gần với tốc độ của pipeline
`generate_vlm_grounded_cot_api.py`. Nếu proxy giới hạn rate, giảm xuống
`--workers 4` hoặc `--workers 8`.

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

## GRPO epoch 2 với rollout CoT log

GRPO giữ nguyên cấu hình cũ:

```text
H100, batch_size=16, num_generations=4, grad_accum=8,
lr=5e-6, kl_beta=0.02, temperature=0.8, max_new_tokens=384
```

Để chạy tiếp từ policy GRPO `checkpoint-grpo-304`, trong khi reference KL vẫn
là SFT `checkpoint-2004`, chạy từ CMD:

```cmd
cd /d D:\du_lieu_that\up_github_1
modal run --detach modal_grpo_qwen9b.py --hf-token hf_xxx --sft-repo KhanhChien/FineTuneQwen35-9B-New --sft-checkpoint checkpoint-2004 --policy-repo KhanhChien/GRPO-Qwen35-9B-New --policy-checkpoint checkpoint-grpo-304 --dataset-repo KhanhChien/Data-Qwen3.5-SFT-New --hub-repo KhanhChien/GRPO-Qwen35-9B-New --epochs 1 --batch-size 16 --num-generations 4 --grad-accum 8 --lr 5e-6 --kl-beta 0.02 --save-steps 50 --epoch-offset 1 --rollout-log-every 1
```

Log được ghi mỗi rollout group vào:

```text
rollouts.log       # dễ đọc bằng mắt: think, answer, reward
rollouts.jsonl     # một dòng cho mỗi generation, dễ phân tích bằng code
```

Sau khi train, hai file được upload vào repo GRPO dưới:

```text
rollout_logs/Qwen3.5-9B-GRPO-checkpoint-grpo-304-epoch2/rollouts.log
rollout_logs/Qwen3.5-9B-GRPO-checkpoint-grpo-304-epoch2/rollouts.jsonl
```

Nếu tên checkpoint epoch 1 khác `checkpoint-grpo-304`, thay đúng giá trị của
`--policy-checkpoint`. `--epochs 1 --epoch-offset 1` là chạy thêm epoch 2,
không phải chạy lại hai epoch từ SFT.