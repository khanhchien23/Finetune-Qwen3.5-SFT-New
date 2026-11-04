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

Script dùng **base Qwen/Qwen3.5-9B + LoRA SFT mới**, không nạp GRPO-304.
Đầu ra mặc định là `Qwen3.5-9B-SFT-Air-Only/final_adapter`. Sau khi SFT xong,
dùng chính adapter này để infer trên các folder `Air.*`.