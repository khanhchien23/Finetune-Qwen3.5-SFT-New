# Stage 1 — Qwen3.5-9B epoch 1 trên validation mô phỏng (Kaggle)

Script: `stage1_qwen9b_val_confusions.py`. **Không train thêm** và **không dùng RealTest**.
Nó đọc đúng split `validation-*.parquet` đã dùng bởi `run_sft_qwen9b.sh`,
chạy Qwen3.5-9B + `checkpoint-668`, rồi thống kê `nhãn dự đoán -> nhãn thật hay nhầm`.

## Chuẩn bị Kaggle

1. Bật **GPU** và **Internet** trong notebook settings; T4 x2 hoặc GPU >=24 GB
   thích hợp hơn cho inference 16-bit. Một T4 đơn lẻ phải offload CPU/disk,
   có thể rất chậm. 8 GB không phù hợp chạy 9B 16-bit; `--load-in-4bit`
   có thể thử riêng nếu đã cài `bitsandbytes` (không bảo đảm vừa 8 GB).
2. Add Input chứa **đúng** dataset Parquet dùng train
   `KhanhChien/Data-Qwen3.5-SFT-New` (thư mục `data/validation-*.parquet`),
   và model Input chứa adapter epoch 1 `checkpoint-668`.
3. Đưa file `stage1_qwen9b_val_confusions.py` vào notebook (upload thành
   Kaggle Dataset riêng hoặc clone repository có file này). Các ô dưới đây
   giả sử đã đặt script tại `/kaggle/working/stage1_qwen9b_val_confusions.py`.

Để tìm đường dẫn **thực tế** của hai Inputs thay vì đoán tên slug, chạy:

```python
from pathlib import Path
val_shards = sorted(Path('/kaggle/input').rglob('validation-*.parquet'))
adapters = sorted(Path('/kaggle/input').rglob('checkpoint-668/adapter_model.safetensors'))
print('Validation shards:', *val_shards, sep='\n  ')
print('Epoch-1 adapters:', *adapters, sep='\n  ')
assert len(val_shards) == 1, 'Chọn đúng dataset Parquet được dùng cho run SFT 9B.'
assert len(adapters) == 1, 'Chọn đúng checkpoint-668 epoch 1.'
PARQUET_DIR = val_shards[0].parent.parent  # parent = data/, parent.parent = dataset root
ADAPTER = adapters[0].parent
print('PARQUET_DIR =', PARQUET_DIR)
print('ADAPTER =', ADAPTER)
```

Nếu script nằm trong Input, sao chép script ra `/kaggle/working/` (thay đường
dẫn nguồn theo đường dẫn Kaggle thực tế):

```python
from pathlib import Path
from shutil import copyfile
scripts = sorted(Path('/kaggle/input').rglob('stage1_qwen9b_val_confusions.py'))
assert len(scripts) == 1, scripts
copyfile(scripts[0], '/kaggle/working/stage1_qwen9b_val_confusions.py')
```

Kiểm tra dependencies; cài thêm **chỉ nếu thiếu**. Dùng các phiên bản
`transformers`, `peft`, `datasets`, `accelerate`, `pyarrow`, `Pillow`, `torch`
giống notebook inference Qwen đã chạy của bạn; chế độ 4-bit cần `bitsandbytes`.
Không cần FAISS/encoder retrieval ở giai đoạn 1.

```python
import importlib.util
for package in ('torch', 'transformers', 'peft', 'datasets', 'pyarrow', 'PIL', 'accelerate'):
    print(package, bool(importlib.util.find_spec(package)))
```

## Chạy thử 5 ảnh rồi chạy đầy đủ

Notebook cell (không cần tự viết các đường dẫn dài thành shell command):

```python
import subprocess, sys
base = [
    sys.executable, '/kaggle/working/stage1_qwen9b_val_confusions.py',
    '--parquet-dir', str(PARQUET_DIR), '--adapter', str(ADAPTER),
    '--output-dir', '/kaggle/working/val_stage1',
]
subprocess.run(base + ['--limit', '5'], check=True)
# Khi 5 ảnh chạy thành công, tiếp tục 1.187 ảnh; tự bỏ qua 5 ảnh đã có.
subprocess.run(base, check=True)
```

Nếu 16-bit bị OOM trên Kaggle, **restart kernel** rồi thử 4-bit bằng một
`--output-dir` **khác**, ví dụ `/kaggle/working/val_stage1_4bit`,
và thêm `--load-in-4bit` vào cả hai lệnh. Không trộn dự đoán 4-bit/16-bit;
chúng có thể khác nhau. Nếu dùng 4-bit ở val nhưng 16-bit trên RealTest, bảng
nhầm lẫn chỉ mang tính tham khảo cho model 16-bit.

Sau khi hoàn tất, output trong `/kaggle/working/val_stage1/` gồm:

- `val_predictions_16bit.csv`: id, file, ground_truth, prediction, correct,
  raw_output. Ghi dần sau từng ảnh; cùng cấu hình có thể resume trong phiên.
- `val_predictions_16bit.run.json`: cấu hình, dấu vân tay adapter và tập val
  để ngăn resume nhầm model/dữ liệu.
- `val_confusion_candidates_16bit.json`: bảng nhầm lẫn **chỉ xuất khi đủ val**.

Trong Kaggle, **Save Version** hoặc tải CSV sau mỗi phiên; nội dung
`/kaggle/working` không tự bảo đảm còn nguyên khi khởi tạo session mới.
Muốn resume qua phiên, đặt lại **cả CSV và `.run.json`** vào cùng output-dir
trước khi chạy. Nếu thiếu file run.json, dùng output-dir mới.

Giai đoạn này chỉ tìm lớp dễ nhầm từ val mô phỏng; chưa tạo kho embedding,
chưa dùng nhãn RealTest và chưa tự sửa dự đoán của Qwen.