"""Inference Qwen3.5-9B + Epoch-1 SFT LoRA (checkpoint-668) on RealTest crops.

Accepts either an extracted RealTest_Full_81Classes directory or
RealTest_Full_81Classes.zip directly (auto-extracts on first run).
Results are appended per batch; rerunning automatically resumes completed rows.
"""

import argparse
import csv
import re
import sys
import time
import zipfile
from pathlib import Path

# Match the training script's torchao / PEFT compatibility workaround.
sys.modules["torchao"] = None
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
import peft.import_utils
peft.import_utils.is_torchao_available = lambda: False
import peft.tuners.lora.torchao
peft.tuners.lora.torchao.is_torchao_available = lambda: False
from peft import PeftModel

MODEL_ID = "Qwen/Qwen3.5-9B"
ROOT = Path(__file__).resolve().parent
DEFAULT_ADAPTER = ROOT / "Qwen3.5-9B-SFT-CoT-81Classes-thinking" / "checkpoint-668"
DEFAULT_REALTEST = ROOT / "RealTest_Full_81Classes.zip"
DEFAULT_OUTPUT = ROOT / "realtest_checkpoint668_predictions.csv"
MIN_PIXELS = 64 * 28 * 28
MAX_PIXELS = 256 * 28 * 28
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
FIELDS = ["file_name", "ground_truth", "prediction", "correct", "raw_output"]

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


def resolve_realtest_dir(data_path: Path) -> Path:
    """Resolve extracted RealTest directory or auto-extract from .zip."""
    dir_candidates = [
        data_path if data_path.is_dir() else data_path.with_suffix(""),
        ROOT / "RealTest_Full_81Classes",
        Path("/workspace/Finetune-Qwen3.5-SFT-New/RealTest_Full_81Classes"),
        Path("/workspace/RealTest_Full_81Classes"),
    ]
    for d in dir_candidates:
        if (d / "crops").is_dir():
            return d.resolve()
        if (d / "RealTest_Full_81Classes" / "crops").is_dir():
            return (d / "RealTest_Full_81Classes").resolve()

    candidates = [
        data_path,
        ROOT / "RealTest_Full_81Classes.zip",
        Path("/workspace/Finetune-Qwen3.5-SFT-New/RealTest_Full_81Classes.zip"),
        Path("/workspace/RealTest_Full_81Classes.zip"),
    ]
    chosen = None
    for cand in candidates:
        if cand.exists():
            chosen = cand.resolve()
            break
    if chosen is None:
        raise FileNotFoundError(f"RealTest data not found at {data_path}")

    if chosen.is_file() and chosen.suffix.lower() == ".zip":
        size_mb = chosen.stat().st_size / (1024 * 1024)
        extract_parent = chosen.parent
        target_dir = extract_parent / chosen.stem
        if not (target_dir / "crops").is_dir():
            print(f"Extracting {chosen} ({size_mb:.1f} MB) -> {extract_parent} ...", flush=True)
            try:
                with zipfile.ZipFile(chosen, "r") as zf:
                    members = [
                        m for m in zf.namelist()
                        if "/crops/" in m or m.endswith("clean_81classes_full_manifest.csv")
                    ]
                    zf.extractall(extract_parent, members=members or None)
            except zipfile.BadZipFile as exc:
                head = chosen.read_bytes()[:120]
                raise RuntimeError(
                    f"{chosen} is not a valid/complete ZIP file (current size: {size_mb:.2f} MB; "
                    f"expected ~1178.3 MB; first bytes: {head!r}). "
                    f"If uploading via SCP/Jupyter/VSCode, wait until upload reaches 1178 MB. "
                    f"If downloading from Google Drive, use `gdown --fuzzy <URL>` instead of wget/curl."
                ) from exc
        chosen = target_dir

    if (chosen / "RealTest_Full_81Classes" / "crops").is_dir():
        chosen = chosen / "RealTest_Full_81Classes"
    return chosen


def realtest_items(root: Path):
    crop_dir = root / "crops"
    if not crop_dir.is_dir():
        raise FileNotFoundError(f"Missing RealTest crops directory: {crop_dir}")
    paths = sorted(p for p in crop_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not paths:
        raise ValueError(f"No image crops found in {crop_dir}")
    manifest = root / "clean_81classes_full_manifest.csv"
    labels = {}
    if manifest.is_file():
        with manifest.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                name = row["crop_file"]
                if name in labels:
                    raise ValueError(f"Duplicate crop_file: {name}")
                labels[name] = row["class_folder_name"]
        if set(labels) != {p.name for p in paths} or any(not label for label in labels.values()):
            raise ValueError(f"Manifest does not match crops: {manifest}")
    else:
        print(f"No manifest: {manifest}; accuracy unavailable", flush=True)
    return paths, labels


def parse_answer(text):
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"<answer>\s*(.*)", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def write_log_entry(stream, row):
    stream.write(
        f"{'=' * 80}\nFile: {row['file_name']}\n"
        f"Ground truth: {row['ground_truth'] or '(unknown)'}\n"
        f"Prediction: {row['prediction'] or '(no answer)'}\nCorrect: {row['correct']}\n"
        f"Raw output:\n{row['raw_output']}\n\n"
    )


def load_processor(adapter: Path):
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
    images, prompts = [], []
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
        images.append(image)
        prompts.append(prompt)
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
            end = next((i + 1 for i, tok in enumerate(ids) if tok in eos_ids), len(ids))
            ids = ids[:end]
        raw = "<think>\n" + processor.tokenizer.decode(ids, skip_special_tokens=False)
        results.append((parse_answer(raw), raw, len(ids)))
    return results



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_REALTEST)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    if not args.adapter.exists() and (args.adapter.parent / "final_adapter").is_dir():
        print(f"Note: {args.adapter} not found; falling back to {args.adapter.parent / 'final_adapter'}", flush=True)
        args.adapter = args.adapter.parent / "final_adapter"
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        if not (args.adapter / filename).is_file():
            raise FileNotFoundError(f"Missing {args.adapter / filename}; check checkpoint-668")
    data_root = resolve_realtest_dir(args.data_root)
    paths, labels = realtest_items(data_root)
    selected = paths[:args.limit] if args.limit else paths
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_output = args.output.with_suffix(".log")
    completed = set()
    if args.output.is_file() and args.output.stat().st_size:
        with args.output.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != FIELDS:
                raise ValueError(f"Unexpected CSV columns: {args.output}")
            completed = {r["file_name"] for r in reader if r["file_name"]}
    remaining = [path for path in selected if path.name not in completed]
    print(f"RealTest: {len(paths)} crops; {len(remaining)} remaining; adapter={args.adapter}", flush=True)
    if not remaining:
        print(f"Complete: {args.output}; log: {log_output}")
        return

    processor = load_processor(args.adapter)
    processor.tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Loading {MODEL_ID} ({dtype}) + {args.adapter}", flush=True)
    base = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=dtype, device_map={"": 0}, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, str(args.adapter), is_trainable=False).eval()
    started = time.monotonic()
    with args.output.open("a", newline="", encoding="utf-8") as stream, log_output.open("a", encoding="utf-8") as log_stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if stream.tell() == 0:
            writer.writeheader(); stream.flush()
        for offset in range(0, len(remaining), args.batch_size):
            batch = remaining[offset:offset + args.batch_size]
            results = infer_batch(batch, processor, model, args.max_new_tokens)
            for idx, (path, (pred, raw, tokens)) in enumerate(zip(batch, results), offset + 1):
                truth = labels.get(path.name, "")
                correct = pred.casefold() == truth.casefold() if truth else ""
                row = {"file_name": path.name, "ground_truth": truth, "prediction": pred, "correct": correct, "raw_output": raw}
                writer.writerow(row); stream.flush()
                write_log_entry(log_stream, row); log_stream.flush()
                print(f"[{idx}/{len(remaining)}] {path.name}: {pred or '(no answer)'} | gt={truth or '?'} | tokens={tokens}", flush=True)
    print(f"Saved {args.output} and {log_output}; elapsed {time.monotonic()-started:.0f}s", flush=True)
    if labels:
        with args.output.open(newline="", encoding="utf-8") as stream:
            rows = [r for r in csv.DictReader(stream) if r["file_name"] in {p.name for p in selected}]
        count = sum(r["correct"] == "True" for r in rows)
        print(f"Accuracy on {len(rows)} crops: {count}/{len(rows)} = {count/len(rows):.2%}", flush=True)


if __name__ == "__main__":
    main()
