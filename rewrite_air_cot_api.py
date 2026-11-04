#!/usr/bin/env python
"""Rewrite the Air-only SFT prompts and CoT using a vision API.

The source Parquet is read directly so the API receives the real embedded
image bytes. The old CoT is provided as context, but the API must rewrite it
for the 30-class Air-only task. Output is resumable JSONL and never overwrites
the source dataset.

Required environment variable:
    OPENAI_API_KEY=...

Example:
    python rewrite_air_cot_api.py --dataset-dir D:\\du_lieu_that\\hf_sft_air_only_parquet \
      --output-dir D:\\du_lieu_that\\hf_sft_air_only_v2 --limit 5
"""

import argparse
import base64
import concurrent.futures
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from datasets import Image as HFImage
from datasets import load_dataset


AIR_CLASSES = [
    "Air.A10", "Air.AH64", "Air.AS365", "Air.B2", "Air.B29", "Air.C130",
    "Air.C17", "Air.Casa212", "Air.CH47", "Air.CH53", "Air.F22", "Air.F35",
    "Air.IL76", "Air.J7", "Air.JH7", "Air.Ka27", "Air.Ka52", "Air.KC135",
    "Air.L39", "Air.Mi24", "Air.Mi28", "Air.Mi8", "Air.Mig29", "Air.Rafale",
    "Air.Su22", "Air.Su27", "Air.T6Texan", "Air.Yak130", "Air.Yak52", "Air.Z8",
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
USER_PROMPT = """Examine the morphological features and viewing geometry of the military platform shown here. Write your step-by-step reasoning in <think>...</think> and state the final class from the 30 Air-only categories in <answer>...</answer>."""
REWRITE_INSTRUCTION = """Rewrite the supplied reasoning for the Air-only 30-class task while preserving the original reasoning format and writing style. Keep the four original sections: Viewpoint & Environment, Coarse Category, Fine-Grained Visual Features, and Disambiguation & Conclusion. Preserve the correct target label exactly. Remove only references to the old 81-class inventory and non-Air categories such as ground vehicles or naval vessels. Do not invent visual details that are unsupported by the image. Return JSON with exactly one field named assistant_cot."""


def image_bytes(image):
    if isinstance(image, dict):
        raw = image.get("bytes")
        if raw:
            return raw, image.get("path", "image.png")
    import io
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue(), "image.png"


def extract_answer(cot):
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", cot, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def validate_cot(cot, label):
    if not isinstance(cot, str) or not cot.startswith("<think>"):
        raise ValueError("assistant_cot must start with <think>")
    if "</think>" not in cot or "<answer>" not in cot:
        raise ValueError("assistant_cot is missing think/answer tags")
    if extract_answer(cot) != label:
        raise ValueError(f"answer {extract_answer(cot)!r} does not equal gt_class {label!r}")
    lowered = cot.lower()
    forbidden = ("grd.", "sea.", "81 target", "81 classes", "ground vehicle", "naval vessel")
    found = [term for term in forbidden if term in lowered]
    if found:
        raise ValueError(f"old/non-aircraft references remain: {found}")


def api_call(api_key, base_url, model, image, old_cot, label, timeout, retries):
    raw, filename = image_bytes(image)
    mime = "image/jpeg" if str(filename).lower().endswith((".jpg", ".jpeg")) else "image/png"
    image_data = f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
    prompt = f"{REWRITE_INSTRUCTION}\n\nGround-truth label (must be preserved exactly): {label}\n\nOld CoT:\n{old_cot}"
    payload = {
        "model": model,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data, "detail": "high"}},
            ]},
        ],
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    last_error = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                result = {"assistant_cot": content}
            cot = result.get("assistant_cot", "")
            validate_cot(cot, label)
            return cot
        except (urllib.error.HTTPError, urllib.error.URLError, KeyError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep((2 ** attempt) + random.random())
    raise RuntimeError(f"API/rewrite failed after {retries + 1} attempts: {last_error}")


def process_split(dataset_dir, split, output, args, api_key):
    files = sorted(dataset_dir.rglob(f"{split}*.parquet"))
    if not files:
        raise FileNotFoundError(f"No {split}*.parquet found below {dataset_dir}")
    dataset = load_dataset("parquet", data_files={split: [str(p) for p in files]})[split]
    dataset = dataset.cast_column("image", HFImage(decode=True))
    count = min(len(dataset), args.limit) if args.limit else len(dataset)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if output.is_file():
        with output.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    done.add(json.loads(line)["id"])
    jobs = []
    for index in range(count):
        job_id = f"{split}:{index}"
        if job_id in done:
            continue
        row = dataset[index]
        jobs.append((job_id, index, row))
    print(f"{split}: total={count}, already_done={len(done)}, remaining={len(jobs)}", flush=True)
    file_lock = threading.Lock()

    def work(job):
        job_id, index, row = job
        label = row.get("gt_class", "")
        if label not in AIR_CLASSES:
            raise ValueError(f"Unexpected gt_class at {job_id}: {label}")
        old_cot = row.get("assistant_cot", "")
        if not old_cot and isinstance(row.get("messages"), list):
            old_cot = row["messages"][-1].get("content", "")
        cot = api_call(api_key, args.base_url, args.model, row["image"], old_cot,
                       label, args.timeout, args.retries)
        record = {
            "id": job_id,
            "split": split,
            "row_index": index,
            "gt_class": label,
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": USER_PROMPT,
            "assistant_cot": cot,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT},
                {"role": "assistant", "content": cot},
            ],
            "image": {"path": row.get("image", {}).get("path", "") if isinstance(row.get("image"), dict) else ""},
        }
        with file_lock, output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
        return job_id

    # The chat-completions vision endpoint accepts one image per request here.
    # batch_size therefore controls how many requests are submitted per wave;
    # workers controls actual concurrent requests within that wave.
    completed_now = 0
    for batch_start in range(0, len(jobs), args.batch_size):
        batch = jobs[batch_start:batch_start + args.batch_size]
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(work, job) for job in batch]
            for future in concurrent.futures.as_completed(futures):
                completed_now += 1
                print(f"{split}: completed {completed_now}/{len(jobs)} "
                      f"(batch {batch_start // args.batch_size + 1}, {future.result()})", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "sk-Z8WrchdOB48aa4L7xtBilW9Bq00HZ1vHR3DkegwXLi04dPXE"))
    parser.add_argument("--base-url", default="https://sv.devquote.shop/v1")
    parser.add_argument("--model", default="claude-opus-5-5")
    parser.add_argument("--split", choices=("train", "validation", "both"), default="both")
    parser.add_argument("--limit", type=int, default=0, help="Rows per split; 0 means all")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Images submitted per API wave; default 16")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=4)
    args = parser.parse_args()
    if not args.api_key:
        raise ValueError("Pass --api-key or set OPENAI_API_KEY")
    if not args.dataset_dir.is_dir() or args.output_dir.resolve() == args.dataset_dir.resolve():
        raise ValueError("dataset directory must exist and output directory must be different")
    if args.limit < 0 or args.batch_size < 1 or args.workers < 1:
        raise ValueError("limit must be >= 0; batch_size and workers must be >= 1")
    if args.workers > args.batch_size:
        raise ValueError("workers cannot be greater than batch_size")
    splits = ("train", "validation") if args.split == "both" else (args.split,)
    for split in splits:
        process_split(args.dataset_dir, split, args.output_dir / f"{split}-air-only-v2.jsonl", args, args.api_key)


if __name__ == "__main__":
    main()