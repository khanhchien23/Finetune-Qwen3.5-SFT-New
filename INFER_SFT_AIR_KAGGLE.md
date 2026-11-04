# Infer SFT Air-only trên Kaggle và ghi JSONL

Script:

```text
inference_qwen9b_sft_air_kaggle.py
```

Script dùng `Qwen/Qwen3.5-9B` + adapter:

```text
KhanhChien/FineTuneQwen35-9B-Air-Only/final_adapter
```

Dataset ảnh có dạng:

```text
/kaggle/input/.../data_that/selected_no_errors/Air.A10/image.jpg
/kaggle/input/.../data_that/selected_no_errors/Air.F35/image.jpg
```

Tên folder chỉ được dùng làm `ground_truth` **sau khi** model dự đoán xong.
Nó không được đưa vào prompt/model input.

## Chạy

Đưa cả hai file Python vào `/kaggle/working/`:

```text
inference_qwen9b_sft_air_kaggle.py
inference_qwen9b_grpo_kaggle.py
```

Script SFT tái sử dụng loader từ file thứ hai. Bật GPU, bật Internet và thêm
Kaggle Secret `HF_TOKEN` có quyền đọc repo SFT, sau đó chạy:

```python
%run /kaggle/working/inference_qwen9b_sft_air_kaggle.py
```

Kết quả được ghi từng dòng ngay vào:

```text
/kaggle/working/air_sft_predictions.jsonl
```

Mỗi dòng là một JSON object dễ đọc bằng mắt, ví dụ có các trường:

```json
{
  "file_name": "Air.A10/img00596_ann00893.jpg",
  "ground_truth": "Air.A10",
  "prediction": "Air.A10",
  "correct": true,
  "generated_tokens": 210,
  "raw_output": "<think>...</think><answer>Air.A10</answer>"
}
```

Nếu Kaggle ngắt, chạy lại sẽ đọc các dòng cũ và bỏ qua ảnh đã có. Muốn thử
trước 5 ảnh, sửa `LIMIT = 5`; chạy toàn bộ dùng `LIMIT = 0`. Có thể chia phiên
bằng `START_INDEX` và `STOP_INDEX`. Tải file JSONL về máy để xem bằng VS Code,
Notepad++ hoặc Python.