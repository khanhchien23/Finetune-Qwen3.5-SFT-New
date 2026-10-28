"""Inference Qwen3.5-9B + SFT LoRA (checkpoint or final adapter) on RealTest crops.

Requires a completed final_adapter, RealTest crops extracted from the uploaded ZIP,
and a free GPU. Results are appended per image; rerunning skips completed rows.
"""

import argparse
import csv
import re
import sys
import time
from pathlib import Path

# Match the training script's torchao / PEFT compatibility workaround.
sys.modules["torchao"] = None

Cline wants to execute this command:
Completed
python -m py_compile 'D:\du_lieu_that\up_github\inference_qwen9b_lora_vm.py'; if ($LASTEXITCODE -ne 0) { throw 'syntax failed' }; git -C 'D:\du_lieu_that\up_github' diff --check; git -C 'D:\du_lieu_that\up_github' status --short
M run_sft_qwen9b.sh
 M train_sft_qwen4b_81classes.py
?? inference_qwen9b_lora_vm.py
Cline read 2 files:

up_github/inference_qwen9b_lora_vm.py · lines 115-174‎

up_github/inference_qwen9b_lora_vm.py · lines 194-260‎


Cú pháp đã qua. Mình sẽ kiểm tra thêm bằng mock hai ảnh có độ dài prompt khác nhau: đảm bảo generate cắt theo độ rộng chuỗi đã padding, mỗi ảnh nhận đúng <answer> và token sau EOS không bị ghi vào log.


Cline wants to execute this command:
Completed
@'
import ast, pathlib, sys, types
p=pathlib.Path(r'D:\du_lieu_that\up_github\inference_qwen9b_lora_vm.py'); tree=ast.parse(p.read_text(encoding='utf-8')); names={'infer_batch','parse_answer'}; funcs=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
class Image:
 def __enter__(self): return self
 def __exit__(self,*args): return False
 def convert(self,mode): return self
class Batch(dict):
 def to(self,device): return self
class Tensor:
 def __init__(self,rows): self.rows=rows; self.shape=(len(rows),len(rows[0]))
 def __getitem__(self,key):
  rows, cols=key; return Tensor([r[cols] for r in self.rows])
 def tolist(self): return self.rows
class Processor:
 def __init__(self): self.tokenizer=types.SimpleNamespace(pad_token_id=0,decode=lambda ids,skip_special_tokens=False: ''.join({1:'<answer>A</answer>',2:'<answer>B</answer>',9:'<|im_end|>'}[i] for i in ids))
 def apply_chat_template(self,*a,**k): return '<|im_start|>assistant\n<think>\n'
 def __call__(self,**kwargs):
  assert len(kwargs['images'])==2 and kwargs['padding'] is True
  return Batch(input_ids=Tensor([[0,3,4],[3,4,5]]))
class Model:
 generation_config=types.SimpleNamespace(eos_token_id=9)
 def generate(self,**kwargs): return Tensor([[0,3,4,1,9,0],[3,4,5,2,9,0]])
ns={'Image':types.SimpleNamespace(open=lambda path:Image()),'torch':types.SimpleNamespace(inference_mode=lambda:types.SimpleNamespace(__enter__=lambda s:None,__exit__=lambda s,*a:False)),'SYSTEM_PROMPT':'s','USER_PROMPT':'u','re':__import__('re')}
exec(compile(ast.Module(body=funcs,type_ignores=[]),str(p),'exec'),ns)
result=ns['infer_batch'](['one.jpg','two.jpg'],Processor(),Model(),16)
assert [r[0] for r in result]==['A','B'] and [r[2] for r in result]==[2,2] and '0' not in result[0][1]
print('batched decoding / EOS / per-image predictions OK')
'@ | python -; if ($LASTEXITCODE -ne 0) { throw 'mock batch failed' }
[Command exited with code 1]
Traceback (most recent call last):
  File "<stdin>", line 25, in <module>
  File "D:\du_lieu_that\up_github\inference_qwen9b_lora_vm.py", line 152, in infer_batch
    with torch.inference_mode():
AttributeError: __enter__
mock batch failed
At line:29 char:43
+ '@ | python -; if ($LASTEXITCODE -ne 0) { throw 'mock batch failed' }
+                                           ~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : OperationStopped: (mock batch failed:String) [], RuntimeException
    + FullyQualifiedErrorId : mock batch failed


Bài test mock vừa vấp lỗi ở context manager giả của test (__enter__), không phải traceback từ PyTorch thật. Mình sẽ sửa mock và chạy lại để xác nhận logic batch/CSV trước khi chốt lệnh benchmark trên máy ảo.


Cline wants to execute this command:
Completed
@'
import ast, pathlib, types
p=pathlib.Path(r'D:\du_lieu_that\up_github\inference_qwen9b_lora_vm.py'); tree=ast.parse(p.read_text(encoding='utf-8')); funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('infer_batch','parse_answer')]
class Image:
 def __enter__(self): return self
 def __exit__(self,*a): return False
 def convert(self,mode): return self
class Context:
 def __enter__(self): return None
 def __exit__(self,*a): return False
class Tensor:
 def __init__(self,rows): self.rows=rows; self.shape=(len(rows),len(rows[0]))
 def __getitem__(self,key): return Tensor([row[key[1]] for row in self.rows])
 def tolist(self): return self.rows
class Batch(dict):
 def to(self,device): return self
class Processor:
 def __init__(self): self.tokenizer=types.SimpleNamespace(pad_token_id=0,decode=lambda ids,skip_special_tokens=False: ''.join({1:'<answer>A</answer>',2:'<answer>B</answer>',9:'<|im_end|>'}[i] for i in ids))
 def apply_chat_template(self,*a,**k): return '<|im_start|>assistant\n<think>\n'
 def __call__(self,**kwargs):
  assert len(kwargs['images'])==2 and kwargs['padding'] is True
  return Batch(input_ids=Tensor([[0,3,4],[3,4,5]]))
class Model:
 generation_config=types.SimpleNamespace(eos_token_id=9)
 def generate(self,**kwargs): return Tensor([[0,3,4,1,9,0],[3,4,5,2,9,0]])
ns={'Image':types.SimpleNamespace(open=lambda path:Image()),'torch':types.SimpleNamespace(inference_mode=lambda:Context()),'SYSTEM_PROMPT':'s','USER_PROMPT':'u','re':__import__('re')}
exec(compile(ast.Module(body=funcs,type_ignores=[]),str(p),'exec'),ns)
result=ns['infer_batch'](['one.jpg','two.jpg'],Processor(),Model(),16)
assert [r[0] for r in result]==['A','B'] and [r[2] for r in result]==[2,2] and '<|im_end|>' in result[0][1] and not result[0][1].endswith('0'); print('batched decoding / EOS / per-image predictions OK')
'@ | python -; if ($LASTEXITCODE -ne 0) { throw 'mock batch failed' }; git -C 'D:\du_lieu_that\up_github' diff --check
batched decoding / EOS / per-image predictions OK

Completed

Có thể thử batch inference, nhưng 18,5/31,9 GB VRAM không có nghĩa là có thể tăng batch tùy ý. GPU đang tải 86%; batch lớn có thể OOM hoặc thậm chí chậm hơn vì ảnh và câu trả lời có độ dài khác nhau.

Mình đã bổ sung --batch-size vào D:\du_lieu_that\up_github\inference_qwen9b_lora_vm.py. Mặc định vẫn là 1; bản mới dùng left-padding cho batch và ghi kết quả từng ảnh vào CSV/log như trước. Mình đã kiểm tra cú pháp và thử logic tách câu trả lời bằng mock; chưa benchmark trên GPU máy ảo.

Nếu inference hiện tại vẫn đang chạy, đừng upload đè script hoặc khởi chạy thêm tiến trình. Sau khi dừng hoặc hoàn tất, làm như sau:

Upload lại file Windows D:\du_lieu_that\up_github\inference_qwen9b_lora_vm.py vào /root/Finetune-Qwen3.5-SFT/. Kiểm tra trên máy ảo:


python3 -m py_compile "$HOME/Finetune-Qwen3.5-SFT/inference_qwen9b_lora_vm.py"
python3 "$HOME/Finetune-Qwen3.5-SFT/inference_qwen9b_lora_vm.py" --help | grep batch-size
Benchmark trước với batch 2, dùng CSV riêng và cùng adapter để không trộn kết quả. Ví dụ đang test final_adapter:


cd "$HOME/Finetune-Qwen3.5-SFT" || exit 1
time python3 inference_qwen9b_lora_vm.py \
  --adapter "$PWD/Qwen3.5-9B-SFT-CoT-81Classes-thinking/final_adapter" \
  --data-root /workspace/RealTest_Full_81Classes \
  --output "$PWD/realtest_final_adapter_batch2_test.csv" \
  --limit 10 \
  --batch-size 2
So thời gian processed 10 in ...s với 75 giây/10 ảnh ở batch 1 mà bạn từng đo. Lưu ý lần đo 75 giây là checkpoint-4008, nên nếu batch 2 chạy final_adapter, đây chỉ là tham khảo; để so sánh công bằng, dùng cùng adapter và cùng 10 ảnh. Kiểm tra cả prediction và raw_output vì padding có thể làm kết quả khác.

Nếu batch 2 không OOM và thực sự nhanh hơn, chạy tiếp cùng CSV bằng --limit 0 --batch-size 2. Chỉ thử batch 4 sau khi đã đo batch 2; không nên nhảy thẳng lên 8 trên GPU 31,9 GB.

Kết quả batch 2 được lưu riêng, nên CSV cũ của checkpoint-4008 hoặc final_adapter không bị xoá. Nếu phát sinh OOM, để tiến trình lỗi thoát và kiểm tra nvidia-smi trước khi chạy lại với --batch-size 1; các dòng CSV đã ghi vẫn còn để tiếp tục.



chưa kill được tiến trình


Mình sẽ chỉ cách xác định lại PID inference hiện tại và dừng đúng tiến trình đó từ terminal SSH thứ hai. Không nên dùng PID cũ hay chạy thêm batch khi tiến trình cũ còn giữ GPU.import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
import peft.import_utils
peft.import_utils.is_torchao_available = lambda: False
import peft.tuners.lora.torchao
peft.tuners.lora.torchao.is_torchao_available = lambda: False
from peft import PeftModel

MODEL_ID = "Qwen/Qwen3.5-9B"
DEFAULT_ADAPTER = Path.home() / "Finetune-Qwen3.5-SFT" / "Qwen3.5-9B-SFT-CoT-81Classes-thinking" / "final_adapter"
DEFAULT_REALTEST = Path("/workspace/RealTest_Full_81Classes/RealTest_Full_81Classes")
DEFAULT_OUTPUT = Path.home() / "Finetune-Qwen3.5-SFT" / "realtest_final_adapter_predictions.csv"
MIN_PIXELS = 64 * 28 * 28
MAX_PIXELS = 256 * 28 * 28
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
FIELDS = ["file_name", "ground_truth", "prediction", "correct", "raw_output"]

# Exact first prompt variant used by the SFT training Parquet and existing Kaggle inference.
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


def realtest_items(root):
    crops = root / "crops"
    if not crops.is_dir():
        raise FileNotFoundError(f"Missing crops directory: {crops}")
    paths = sorted(p for p in crops.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not paths:
        raise ValueError(f"No crop images in {crops}")
    manifest = root / "clean_81classes_full_manifest.csv"
    labels = {}
    if manifest.is_file():
        with manifest.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                name = row["crop_file"]
                if name in labels:
                    raise ValueError(f"Duplicate crop_file in manifest: {name}")
                labels[name] = row["class_folder_name"]
        if set(labels) != {p.name for p in paths} or any(not label for label in labels.values()):
            raise ValueError(f"Manifest does not match crops: {manifest}")
    else:
        print(f"No manifest at {manifest}; no ground truth/accuracy available", flush=True)
    return paths, labels


def parse_answer(raw):
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"<answer>\s*(.*)", raw, re.DOTALL)
    return match.group(1).strip() if match else ""


def write_log_entry(stream, row):
    stream.write(
        f"{'=' * 80}\n"
        f"File: {row['file_name']}\n"
        f"Ground truth: {row['ground_truth'] or '(unavailable)'}\n"
        f"Predicted label: {row['prediction'] or '(no <answer> found)'}\n"
        f"Correct: {row['correct'] if row['ground_truth'] else '(unavailable)'}\n"
        f"Model output (<think> and <answer>):\n{row['raw_output']}\n\n"
    )


def load_processor(adapter):
    source = str(adapter) if (adapter / "processor_config.json").is_file() else MODEL_ID
    try:
        return AutoProcessor.from_pretrained(
            source, size={"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS},
            trust_remote_code=True,
        )
    except (TypeError, ValueError):
        return AutoProcessor.from_pretrained(
            source, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS, trust_remote_code=True,
        )


def infer_batch(paths, processor, model, max_new_tokens):
    images = []
    prompts = []
    for path in paths:
        with Image.open(path) as source:
            image = source.convert("RGB")
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": USER_PROMPT},
            ]},
        ]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        if not prompt.endswith("<|im_start|>assistant\n<think>\n"):
            raise ValueError("Chat template does not match the SFT thinking prompt")
        images.append(image)
        prompts.append(prompt)
    # Left-pad for batched generation. No truncation: image tokens must stay
    # aligned with pixel values; generated sequences share the padded width.
    inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    new_ids = generated[:, inputs["input_ids"].shape[1]:]
    eos_id = model.generation_config.eos_token_id
    eos_ids = {eos_id} if isinstance(eos_id, int) else set(eos_id or [])
    results = []
    for ids in new_ids.tolist():
        if eos_ids:
            end = next((index + 1 for index, token in enumerate(ids) if token in eos_ids), len(ids))
            ids = ids[:end]
        raw = "<think>\n" + processor.tokenizer.decode(ids, skip_special_tokens=False)
        results.append((parse_answer(raw), raw, len(ids)))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_REALTEST,
                        help="Directory containing crops/ and optional clean_81classes_full_manifest.csv")
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log-output", type=Path, default=None,
                        help="Readable per-image thinking/answer log; default: --output with .log suffix")
    parser.add_argument("--limit", type=int, default=5, help="First N crops; 0 = all")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Images per generate call; try 2 then 4 after checking VRAM")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    if args.limit < 0 or args.max_new_tokens < 1 or args.batch_size < 1:
        parser.error("--limit must be >= 0, --batch-size and --max-new-tokens must be >= 1")
    if not (args.adapter / "adapter_config.json").is_file() or not (
        args.adapter / "adapter_model.safetensors"
    ).is_file():
        raise FileNotFoundError(f"Adapter files not found: {args.adapter}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for 9B 16-bit inference")
    paths, labels = realtest_items(args.data_root)
    selected = paths[:args.limit] if args.limit else paths
    log_output = args.log_output if args.log_output is not None else args.output.with_suffix(".log")
    if log_output.resolve() == args.output.resolve():
        raise ValueError("Log output must differ from CSV output")

    completed = set()
    if args.output.is_file() and args.output.stat().st_size:
        with args.output.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != FIELDS:
                raise ValueError(f"Output CSV has unexpected columns: {args.output}")
            completed = {row["file_name"] for row in reader if row["file_name"]}
    # Rebuild the readable log from CSV before inferring: interrupted runs and
    # results generated with older versions are included without rerunning images.
    log_output.parent.mkdir(parents=True, exist_ok=True)
    with log_output.open("w", encoding="utf-8") as log_stream:
        if args.output.is_file() and args.output.stat().st_size:
            with args.output.open(newline="", encoding="utf-8") as csv_stream:
                for row in csv.DictReader(csv_stream):
                    if row["file_name"]:
                        write_log_entry(log_stream, row)
    remaining = [path for path in selected if path.name not in completed]
    print(f"RealTest: {len(paths)} crops, {len(remaining)} remaining of {len(selected)} selected", flush=True)
    if not remaining:
        print(f"Readable log: {log_output}", flush=True)
        return

    processor = load_processor(args.adapter)
    processor.tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Loading {MODEL_ID} in {dtype} onto GPU 0; adapter: {args.adapter}", flush=True)
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=dtype, device_map={"": 0}, trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(args.adapter), is_trainable=False).eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    start = time.monotonic()
    with args.output.open("a", newline="", encoding="utf-8") as stream, log_output.open(
        "a", encoding="utf-8"
    ) as log_stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if stream.tell() == 0:
            writer.writeheader()
            stream.flush()
        for start_index in range(0, len(remaining), args.batch_size):
            batch_paths = remaining[start_index:start_index + args.batch_size]
            results = infer_batch(batch_paths, processor, model, args.max_new_tokens)
            for index, (path, (prediction, raw, count)) in enumerate(
                zip(batch_paths, results), start_index + 1
            ):
                gt = labels.get(path.name, "")
                match = prediction.casefold() == gt.casefold() if gt else ""
                correct += bool(match)
                row = dict(file_name=path.name, ground_truth=gt,
                           prediction=prediction, correct=match, raw_output=raw)
                writer.writerow(row)
                stream.flush()
                write_log_entry(log_stream, row)
                log_stream.flush()
                print(f"[{index}/{len(remaining)}] {path.name}: {prediction or '(no answer)'} "
                      f"| gt={gt or '(unknown)'} | tokens={count}/{args.max_new_tokens}", flush=True)
    print(f"Saved {args.output} and {log_output}; processed {len(remaining)} "
          f"in {time.monotonic()-start:.0f}s", flush=True)
    if labels:
        with args.output.open(newline="", encoding="utf-8") as stream:
            rows = [row for row in csv.DictReader(stream) if row["file_name"] in {p.name for p in selected}]
        print(f"Accuracy (selected crops): {sum(row['correct'] == 'True' for row in rows)}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()