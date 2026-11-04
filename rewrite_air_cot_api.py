#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fast async Air-only CoT rewrite, structured like generate_vlm_grounded_cot_api.py."""

import argparse
import asyncio
import base64
import io
import json
import os
import random
import re
import time
from pathlib import Path

import openai
from datasets import Image as HFImage
from datasets import load_dataset

AIR_CLASSES = [
    "Air.A10", "Air.AH64", "Air.AS365", "Air.B2", "Air.B29", "Air.C130", "Air.C17",
    "Air.Casa212", "Air.CH47", "Air.CH53", "Air.F22", "Air.F35", "Air.IL76", "Air.J7",
    "Air.JH7", "Air.Ka27", "Air.Ka52", "Air.KC135", "Air.L39", "Air.Mi24", "Air.Mi28",
    "Air.Mi8", "Air.Mig29", "Air.Rafale", "Air.Su22", "Air.Su27", "Air.T6Texan",
    "Air.Yak130", "Air.Yak52", "Air.Z8",
]
AIR_CLASS_TEXT = ", ".join(AIR_CLASSES)

SYSTEM_PROMPT = f"""You are an expert military analyst specializing in Fine-Grained Military Vehicle Recognition.
Your task is to inspect the provided image and classify the military platform into EXACTLY ONE of the following 30 Air-only target classes:

{AIR_CLASS_TEXT}

Follow this structured reasoning process before making your final decision:
1. Viewpoint & Environment: Identify the camera viewing angle and operational setting.
2. Coarse Category: Determine the primary aircraft domain (Air) and general aircraft type.
3. Fine-Grained Visual Features: Describe the distinctive physical characteristics strictly visible in the image.
4. Disambiguation & Conclusion: Rule out visually similar aircraft types based on observed features and select the exact target class name.

Required output format:
<think>
Write your step-by-step visual analysis here...
</think>
<answer>Exact_Air_Class_Name</answer>"""
USER_PROMPT = (
    "Examine the morphological features and viewing geometry of the military platform shown here. "
    "Write your step-by-step reasoning in <think>...</think> and state the final class from "
    "the 30 Air-only categories in <answer>...</answer>."
)
REWRITE_INSTRUCTION = (
    "Rewrite the supplied reasoning for the Air-only 30-class task while preserving the original "
    "reasoning format and writing style. Keep the four original sections: Viewpoint & Environment, "
    "Coarse Category, Fine-Grained Visual Features, and Disambiguation & Conclusion. Preserve the "
    "correct target label exactly. Remove only references to the old 81-class inventory and non-Air "
    "categories such as ground vehicles or naval vessels. Do not invent visual details unsupported "
    "by the image. In Steps 1, 2, and 3, never write any Air.* class label, "
    "the ground-truth label, or meta phrases such as target classification, validated, "
    "standard processing, or confirmed. Mention candidate class labels only in Step 4. "
    "Output ONLY the exact <think>...</think>\\n<answer>...</answer> block."
)


def image_to_base64(image):
    if isinstance(image, dict) and image.get("bytes"):
        raw = image["bytes"]
    else:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=90, optimize=True)
        raw = buffer.getvalue()
    return base64.b64encode(raw).decode("ascii")


def text_content(content):
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(x.get("text", "") for x in content if isinstance(x, dict)).strip()
    return str(content).strip()


def extract_answer(cot):
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", cot, re.I | re.S)
    return match.group(1).strip() if match else ""


def validate_cot(cot, label):
    if not isinstance(cot, str) or not re.search(r"<think>.*?</think>", cot, re.I | re.S):
        return False, "missing think tags"
    if extract_answer(cot) != label:
        return False, f"answer {extract_answer(cot)!r} != {label!r}"
    think_body = re.search(r"<think>\s*(.*?)\s*</think>", cot, re.I | re.S).group(1)
    headings = [
        re.search(r"1\.\s*Viewpoint(?:\s*&\s*Environment)?\s*:", think_body, re.I),
        re.search(r"2\.\s*Coarse Category\s*:", think_body, re.I),
        re.search(r"3\.\s*Fine-Grained Visual Features\s*:", think_body, re.I),
        re.search(r"4\.\s*Disambiguation\s*&\s*Conclusion\s*:", think_body, re.I),
    ]
    if not all(headings):
        return False, "missing numbered reasoning section"
    step1_3 = think_body[:headings[3].start()]
    early_labels = re.findall(r"\bAir\.[A-Za-z0-9.]+\b", step1_3)
    if early_labels:
        return False, f"Air class label appears before Step 4: {sorted(set(early_labels))}"
    forbidden = (
        "grd.", "sea.", "81 target", "81 classes", "ground vehicle", "naval vessel",
        "target classification", "ground-truth", "standard processing", "validated",
        "validation passed", "jumping directly", "template applied",
    )
    found = [word for word in forbidden if word in cot.lower()]
    return (False, f"forbidden references {found}") if found else (True, "OK")


def old_cot(row):
    if row.get("assistant_cot"):
        return row["assistant_cot"]
    messages = row.get("messages")
    return text_content(messages[-1].get("content", "")) if isinstance(messages, list) and messages else ""


def redact_class_labels(cot):
    """Prevent the old answer label from leaking into the new visual reasoning."""
    for label in sorted(AIR_CLASSES, key=len, reverse=True):
        cot = re.sub(rf"\b{re.escape(label)}\b", "the target aircraft class", cot,
                     flags=re.IGNORECASE)
    return cot


async def rewrite_one(client, model, row, record_id, semaphore, timeout, retries):
    label = row.get("gt_class", "")
    if label not in AIR_CLASSES:
        raise ValueError(f"{record_id}: unexpected gt_class {label!r}")
    request_text = (
        f"{REWRITE_INSTRUCTION}\n\n"
        f"The answer MUST be exactly `<answer>{label}</answer>`. Do not write `{label}` "
        "anywhere before the <answer> tag.\n\n"
        f"Old visual reasoning to rewrite (class labels have been redacted):\n"
        f"{redact_class_labels(old_cot(row))}"
    )
    async with semaphore:
        last_error = ""
        for attempt in range(retries + 1):
            try:
                response = await asyncio.wait_for(client.chat.completions.create(
                    model=model,
                    temperature=0.2,
                    max_tokens=500,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": [
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/jpeg;base64,{image_to_base64(row['image'])}"
                            }},
                            {"type": "text", "text": request_text},
                        ]},
                    ],
                ), timeout=timeout)
                cot = text_content(response.choices[0].message.content)
                valid, reason = validate_cot(cot, label)
                if valid:
                    return cot
                last_error = reason
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                request_text += (
                    f"\n\nPrevious attempt was rejected because: {last_error}. "
                    "Rewrite it again and fix this exact problem."
                )
                await asyncio.sleep(1.0 + attempt * 1.5 + random.random())
    raise RuntimeError(f"{record_id} failed after {retries + 1} attempts: {last_error}")


def load_existing(output):
    completed = {}
    if not output.is_file():
        return completed
    for line in output.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if validate_cot(record.get("assistant_cot", ""), record.get("gt_class", ""))[0]:
                completed[record["id"]] = record
        except (json.JSONDecodeError, KeyError):
            continue
    return completed


def save_output(output, target_rows, completed):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for record_id, _ in target_rows:
            if record_id in completed:
                stream.write(json.dumps(completed[record_id], ensure_ascii=False) + "\n")


async def run_split(args, split, client):
    files = sorted(args.dataset_dir.rglob(f"{split}*.parquet"))
    if not files:
        raise FileNotFoundError(f"No {split}*.parquet below {args.dataset_dir}")
    dataset = load_dataset("parquet", data_files={split: [str(p) for p in files]})[split]
    dataset = dataset.cast_column("image", HFImage(decode=True))
    count = min(len(dataset), args.limit) if args.limit else len(dataset)
    output = args.output_dir / f"{split}-air-only-v2.jsonl"
    completed = load_existing(output)
    target_rows = [(f"{split}:{i}", dataset[i]) for i in range(count)]
    pending = [(record_id, row) for record_id, row in target_rows if record_id not in completed]
    print(f"{split}: target={count} done={len(completed)} remaining={len(pending)} "
          f"batch_size={args.batch_size} concurrency={args.concurrency} model={args.model}", flush=True)
    semaphore = asyncio.Semaphore(args.concurrency)
    started = time.monotonic()
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start + args.batch_size]

        async def worker(pair):
            record_id, row = pair
            cot = await rewrite_one(client, args.model, row, record_id, semaphore,
                                    args.timeout, args.retries)
            return record_id, row, cot

        results = await asyncio.gather(*(worker(pair) for pair in batch), return_exceptions=True)
        failures = [x for x in results if isinstance(x, Exception)]
        for result in results:
            if isinstance(result, Exception):
                print(f"FAILED: {result}", flush=True)
                continue
            record_id, row, cot = result
            completed[record_id] = {
                "id": record_id, "split": split,
                "row_index": int(record_id.split(":")[1]), "gt_class": row["gt_class"],
                "system_prompt": SYSTEM_PROMPT, "user_prompt": USER_PROMPT,
                "assistant_cot": cot,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT},
                    {"role": "assistant", "content": cot},
                ],
                "image": row.get("image", {}).get("path", "") if isinstance(row.get("image"), dict) else "",
            }
        save_output(output, target_rows, completed)
        elapsed = time.monotonic() - started
        print(f"[Batch {start // args.batch_size + 1} | +{len(batch) - len(failures)} "
              f"| saved {len(completed)}/{count} | {len(completed) / max(elapsed, .001):.2f} img/s]",
              flush=True)
        if failures:
            raise RuntimeError(f"{split}: {len(failures)} request(s) failed; rerun to resume") from failures[0]


async def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = openai.AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)
    try:
        splits = ("train", "validation") if args.split == "both" else (args.split,)
        for split in splits:
            await run_split(args, split, client)
    finally:
        await client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--base-url", default="https://sv.devquote.shop/v1")
    parser.add_argument("--model", default="claude-opus-5")
    parser.add_argument("--split", choices=("train", "validation", "both"), default="both")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--concurrency", "--workers", dest="concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    if not args.api_key:
        raise ValueError("Pass --api-key or set OPENAI_API_KEY")
    if not args.dataset_dir.is_dir() or args.output_dir.resolve() == args.dataset_dir.resolve():
        raise ValueError("dataset directory must exist and output directory must differ")
    if args.limit < 0 or args.batch_size < 1 or args.concurrency < 1 or args.retries < 0:
        raise ValueError("limit >= 0; batch-size/concurrency >= 1; retries >= 0")
    if args.concurrency > args.batch_size:
        raise ValueError("concurrency cannot be greater than batch-size")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()