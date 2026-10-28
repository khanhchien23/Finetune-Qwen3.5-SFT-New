"""Stage 1: infer the SFT synthetic validation split and build confusion candidates.

Use the *validation-*.parquet files loaded by run_sft_qwen9b.sh, not train or
RealTest. Run --summarize-only on an existing complete predictions CSV without
loading Qwen. Inference needs Qwen/Qwen3.5-9B + the epoch-1 LoRA checkpoint.
"""

import argparse
import csv
import io
import json
import math
import re
import sys
import hashlib
from collections import Counter, defaultdict
from pathlib import Path


MODEL_ID = "Qwen/Qwen3.5-9B"
DATASET_REPO = "KhanhChien/Data-Qwen3.5-SFT-New"
FIELDS = ["id", "file_name", "ground_truth", "prediction", "correct", "raw_output"]
MIN_PIXELS = 64 * 28 * 28
MAX_PIXELS = 256 * 28 * 28
ANSWER = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
DEFAULT_VAL_JSONL = Path("/kaggle/input/datasets/khanhchien/cot-json/sft_cot_81classes_vlm_val.jsonl")
DEFAULT_IMAGES_DIR = Path("/kaggle/input/datasets/khanhchien/anh-mo-phong-1/images")
DEFAULT_ADAPTER = Path(
    "/kaggle/input/models/khanhchien/model-sft-668/pytorch/default/1/"
    "Qwen3.5-9B-SFT-CoT-81Classes-thinking/checkpoint-668"
)


class JsonlValidationSet:
    """Lightweight dataset wrapper for sft_cot_81classes_val.jsonl + raw images."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in self.rows]
        return self.rows[key]


def resolve_image_path(images_dir, fname, rel_image):
    for candidate in (
        images_dir / fname,
        images_dir / "images" / fname,
        images_dir.parent / rel_image,
        images_dir / rel_image,
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Missing image '{fname}' under {images_dir} (checked {images_dir} and {images_dir / 'images'})"
    )


def load_validation_jsonl(val_jsonl, images_dir):
    """Load validation records from JSONL and match images in images_dir."""
    if not val_jsonl.is_file():
        raise FileNotFoundError(f"Validation JSONL not found: {val_jsonl}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    rows, seen = [], set()
    with val_jsonl.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            sample_id = item["id"]
            if sample_id in seen:
                raise ValueError(f"Duplicate id in validation JSONL: {sample_id}")
            seen.add(sample_id)
            fname = item.get("file_name") or Path(item.get("image", "")).name
            img_path = resolve_image_path(images_dir, fname, item.get("image", ""))
            if "messages" in item:
                raw_msgs = item["messages"]
                sys_text = raw_msgs[0]["content"]
                user_content = raw_msgs[1]["content"]
                if isinstance(user_content, list):
                    user_text = next((c["text"] for c in user_content if c.get("type") == "text"), "")
                else:
                    user_text = str(user_content)
            else:
                sys_text = item["system_prompt"]
                user_text = item["user_prompt"]
            rows.append({
                "id": sample_id,
                "file_name": fname,
                "gt_class": item["gt_class"],
                "bbox": item.get("bbox"),
                "image_path": str(img_path),
                "system_prompt": sys_text,
                "user_prompt": user_text,
            })
    if not rows:
        raise ValueError(f"Empty validation JSONL: {val_jsonl}")
    return JsonlValidationSet(rows)


def load_validation(parquet_dir=None, val_jsonl=None, images_dir=None):
    """Load validation split from JSONL + images_dir or from Parquet shards."""
    if parquet_dir is None:
        return load_validation_jsonl(val_jsonl, images_dir)
    from datasets import load_dataset

    files = sorted((parquet_dir / "data").glob("validation-*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No validation-*.parquet in {parquet_dir / 'data'}; download the "
            f"{DATASET_REPO} dataset or pass --parquet-dir"
        )
    ds = load_dataset("parquet", data_files={"validation": [str(p) for p in files]})["validation"]
    required = {"id", "file_name", "image", "gt_class", "system_prompt", "user_prompt"}
    if not required.issubset(ds.column_names):
        raise ValueError(f"Validation Parquet missing columns: {sorted(required - set(ds.column_names))}")
    ids = ds["id"]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate id in validation split")
    if len(ds) == 0:
        raise ValueError("Empty validation split")
    return ds


def read_predictions(path, known_ids):
    """Validate resumed output before deciding which examples to infer."""
    rows = {}
    if not path.is_file() or path.stat().st_size == 0:
        return rows
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"Unexpected columns in {path}: {reader.fieldnames}")
        for row in reader:
            sample_id = row["id"]
            if sample_id not in known_ids or sample_id in rows:
                raise ValueError(f"Unknown/duplicate validation id: {sample_id}")
            if row["ground_truth"] != known_ids[sample_id]["gt_class"]:
                raise ValueError(f"Ground truth mismatch for {sample_id}")
            if row["file_name"] != known_ids[sample_id]["file_name"]:
                raise ValueError(f"File name mismatch for {sample_id}")
            expected = str(row["prediction"] == row["ground_truth"])
            if row["correct"] != expected:
                raise ValueError(f"Correct flag mismatch for {sample_id}")
            rows[sample_id] = row
    return rows


def parse_answer(raw):
    match = ANSWER.search(raw)
    return match.group(1).strip() if match else ""


def infer_one(item, processor, model, torch, max_new_tokens):
    from PIL import Image

    if "image_path" in item:
        with Image.open(item["image_path"]) as src:
            image = src.convert("RGB")
            bbox = item.get("bbox")
            if bbox and len(bbox) == 4:
                x, y, w, h = [float(v) for v in bbox]
                pad_x, pad_y = w * 0.08, h * 0.08
                x1 = max(0, int(math.floor(x - pad_x)))
                y1 = max(0, int(math.floor(y - pad_y)))
                x2 = min(image.width, int(math.ceil(x + w + pad_x)))
                y2 = min(image.height, int(math.ceil(y + h + pad_y)))
                if x2 > x1 + 4 and y2 > y1 + 4:
                    image = image.crop((x1, y1, x2, y2))
    else:
        image = item["image"]
        if isinstance(image, dict):
            image = Image.open(io.BytesIO(image["bytes"]))
        image = image.convert("RGB")
    messages = [
        {"role": "system", "content": [{"type": "text", "text": item["system_prompt"]}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": item["user_prompt"]},
        ]},
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    if not prompt.endswith("<|im_start|>assistant\n<think>\n"):
        raise ValueError("Chat template does not match the SFT thinking prefix")
    inputs = processor(text=[prompt], images=[image], return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    raw = "<think>\n" + processor.batch_decode(
        generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=False,
    )[0]
    return parse_answer(raw), raw


def load_model(adapter, quantize, offload_dir=None):
    # Same Kaggle PEFT/torchao workaround and loading policy as inference_qwen9b_lora_kaggle.py.
    sys.modules["torchao"] = None
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    import peft.import_utils

    peft.import_utils.is_torchao_available = lambda: False
    import peft.tuners.lora.torchao

    peft.tuners.lora.torchao.is_torchao_available = lambda: False
    from peft import PeftModel

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for Qwen inference; use --summarize-only for CPU")
    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"LoRA adapter not found: {adapter}")
    gpu_count = torch.cuda.device_count()
    memory = {}
    for gpu in range(gpu_count):
        gib = torch.cuda.get_device_properties(gpu).total_memory / 1024**3
        budget = int(gib - 3)
        if budget < 4:
            raise RuntimeError(f"GPU {gpu}: {gib:.1f} GiB insufficient headroom")
        memory[gpu] = f"{budget}GiB"
    memory["cpu"] = "24GiB"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if not quantize and gpu_count == 1 and torch.cuda.get_device_properties(0).total_memory < 20 * 1024**3:
        print("WARNING: 16-bit Qwen 9B will offload weights to CPU/disk on one small GPU; "
              "this can be very slow. --load-in-4bit uses separate output files.", flush=True)
    options = {}
    if quantize:
        from transformers import BitsAndBytesConfig

        options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    source = str(adapter) if (adapter / "processor_config.json").is_file() else MODEL_ID
    try:
        processor = AutoProcessor.from_pretrained(
            source, size={"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS},
            trust_remote_code=True,
        )
    except (TypeError, ValueError):
        processor = AutoProcessor.from_pretrained(
            source, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS, trust_remote_code=True,
        )
    print(f"Loading {MODEL_ID} ({dtype}, 4-bit={quantize}), GPUs={gpu_count}", flush=True)
    if offload_dir is None:
        offload_dir = Path("val_offload")
    offload_dir.mkdir(parents=True, exist_ok=True)
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=dtype, device_map="auto", max_memory=memory,
        offload_folder=str(offload_dir), offload_state_dict=True,
        low_cpu_mem_usage=True, trust_remote_code=True, **options,
    )
    model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False).eval()
    return processor, model, torch


def validate_run_config(path, config, existing_rows):
    """Do not resume an output CSV with a different adapter or generation setup."""
    if path.is_file():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != config:
            raise ValueError(f"Run config mismatch at {path}; use a new --output-dir")
    elif existing_rows:
        raise ValueError(f"Missing {path} for an existing predictions CSV; "
                         "cannot safely resume. Use a new --output-dir")
    else:
        path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def adapter_fingerprint(adapter):
    """Hash the small adapter config + weight file metadata to identify a run."""
    weights = adapter / "adapter_model.safetensors"
    digest = hashlib.sha256()
    digest.update((adapter / "adapter_config.json").read_bytes())
    with weights.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(rows, known_ids, classes, output, max_candidates, min_count):
    """Predicted A -> true B, counting only valid labels; keep A as first candidate."""
    if len(rows) != len(known_ids):
        raise ValueError(f"Incomplete validation predictions: {len(rows)}/{len(known_ids)}")
    total_by_prediction = Counter()
    counts = defaultdict(Counter)
    invalid = Counter()
    correct = 0
    for row in rows.values():
        gt, pred = row["ground_truth"], row["prediction"]
        if pred not in classes:
            invalid[pred or "(empty)"] += 1
            continue
        total_by_prediction[pred] += 1
        counts[pred][gt] += 1
        correct += pred == gt
    candidate_map = {}
    for pred in sorted(classes):
        total = total_by_prediction[pred]
        alternatives = [
            {"label": label, "count": n, "fraction_of_predictions": round(n / total, 6)}
            for label, n in sorted(counts[pred].items(), key=lambda pair: (-pair[1], pair[0]))
            if label != pred and n >= min_count
        ][:max_candidates]
        candidate_map[pred] = {
            "predicted_count": total,
            "correct_count": counts[pred][pred],
            "candidates": [pred] + [item["label"] for item in alternatives],
            "alternatives": alternatives,
        }
    summary = {
        "direction": "predicted_label -> possible true labels",
        "source": "synthetic validation Parquet only (not RealTest)",
        "total": len(known_ids),
        "correct_exact": correct,
        "accuracy_exact": round(correct / len(known_ids), 6),
        "invalid_predictions": dict(sorted(invalid.items())),
        "max_alternatives": max_candidates,
        "min_error_count": min_count,
        "by_prediction": candidate_map,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-jsonl", type=Path, default=DEFAULT_VAL_JSONL,
                        help="Path to sft_cot_81classes_val.jsonl on Kaggle")
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR,
                        help="Path to synthetic images directory on Kaggle")
    parser.add_argument("--parquet-dir", type=Path, default=None,
                        help="Optional Parquet dataset root (if using Parquet instead of JSONL)")
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER,
                        help="Path to checkpoint-668 (epoch 1) LoRA adapter")
    parser.add_argument("--output-dir", type=Path, default=Path("val_stage1"))
    parser.add_argument("--summarize-only", action="store_true", help="Do not load model; require complete CSV")
    parser.add_argument("--load-in-4bit", action="store_true", help="NF4 inference; separate output from 16-bit")
    parser.add_argument("--limit", type=int, default=0, help="Infer first N val rows; 0 = all (partial runs do not summarize)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-candidates", type=int, default=3, help="Maximum alternatives per predicted class")
    parser.add_argument("--min-count", type=int, default=2, help="Minimum val mistakes for an alternative")
    args = parser.parse_args()
    if args.limit < 0 or args.max_new_tokens < 1 or args.max_candidates < 1 or args.min_count < 1:
        parser.error("limit >= 0, max-new-tokens/max-candidates/min-count >= 1 required")
    if not args.summarize_only and args.adapter is None:
        parser.error("--adapter is required unless --summarize-only is specified")
    if not args.summarize_only and not (args.adapter / "adapter_config.json").is_file():
        parser.error(f"Adapter config not found: {args.adapter / 'adapter_config.json'}")
    if not args.summarize_only and not (args.adapter / "adapter_model.safetensors").is_file():
        parser.error(f"Adapter weights not found: {args.adapter / 'adapter_model.safetensors'}")
    ds = load_validation(args.parquet_dir, args.val_jsonl, args.images_dir)
    # Only metadata is materialized here; images are decoded one at a time during inference.
    known_ids = {sample_id: {"gt_class": gt, "file_name": filename}
                 for sample_id, gt, filename in zip(ds["id"], ds["gt_class"], ds["file_name"])}
    classes = set(ds["gt_class"])
    stem = "val_predictions_4bit.csv" if args.load_in_4bit else "val_predictions_16bit.csv"
    predictions = args.output_dir / stem
    candidates = args.output_dir / stem.replace("predictions", "confusion_candidates").replace(".csv", ".json")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_predictions(predictions, known_ids)
    print(f"Validation: {len(ds)} images / {len(classes)} classes; resumed {len(rows)} rows", flush=True)
    if not args.summarize_only:
        validate_run_config(args.output_dir / stem.replace(".csv", ".run.json"), {
            "adapter_sha256": adapter_fingerprint(args.adapter),
            "model_id": MODEL_ID,
            "quantized_4bit": args.load_in_4bit,
            "max_new_tokens": args.max_new_tokens,
            "validation_ids_sha256": hashlib.sha256(
                "\n".join(ds["id"]).encode("utf-8")
            ).hexdigest(),
        }, rows)
        selected = range(min(args.limit, len(ds)) if args.limit else len(ds))
        ids = ds["id"]
        remaining = [i for i in selected if ids[i] not in rows]
        if remaining:
            processor, model, torch = load_model(
                args.adapter, args.load_in_4bit, args.output_dir / "qwen_offload"
            )
            with predictions.open("a", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                if stream.tell() == 0:
                    writer.writeheader()
                    stream.flush()
                for done, i in enumerate(remaining, 1):
                    item = ds[i]
                    pred, raw = infer_one(item, processor, model, torch, args.max_new_tokens)
                    row = {"id": item["id"], "file_name": item["file_name"],
                           "ground_truth": item["gt_class"], "prediction": pred,
                           "correct": pred == item["gt_class"], "raw_output": raw}
                    writer.writerow(row)
                    stream.flush()
                    rows[item["id"]] = row
                    if done % 20 == 0 or done == len(remaining):
                        print(f"Saved {done}/{len(remaining)} remaining -> {predictions}", flush=True)
    if len(rows) != len(ds):
        print(f"Partial CSV: {len(rows)}/{len(ds)}; rerun to resume. No confusion JSON yet.", flush=True)
        return
    summary = summarize(rows, known_ids, classes, candidates, args.max_candidates, args.min_count)
    print(f"Accuracy (exact): {summary['correct_exact']}/{summary['total']} = "
          f"{summary['accuracy_exact']:.2%}; invalid: {sum(summary['invalid_predictions'].values())}")
    print(f"Predictions: {predictions}\nCandidates: {candidates}")


if __name__ == "__main__":
    main()