"""Infer the Air-only SFT adapter on class-folder images in Kaggle.

The model input contains only the image and the fixed classification prompt.
The folder name is used as ground truth only after generation, for evaluation.
Each completed image is appended immediately to a human-readable JSONL file.
"""

import json
import os
import sys
from pathlib import Path

# Reuse the tested Qwen3.5-9B Kaggle loader/prompt/image-folder implementation.
import inference_qwen9b_grpo_kaggle as common


REALTEST_ROOT = Path("/kaggle/input/datasets/khnhchin/data-that/selected_no_errors")
MODEL_ID = "Qwen/Qwen3.5-9B"
SFT_REPO = "KhanhChien/FineTuneQwen35-9B-Air-Only"
SFT_ADAPTER_SUBDIR = "final_adapter"
ADAPTER_INPUT = None  # Set to an attached Kaggle Model folder to avoid HF download.
ADAPTER = None
OUTPUT = Path("/kaggle/working/air_sft_predictions.jsonl")
LIMIT = 0
START_INDEX = 0
STOP_INDEX = 0
MAX_NEW_TOKENS = 384
LOAD_IN_4BIT = False
ATTENTION_IMPLEMENTATION = None


def resolve_adapter():
    """Find a local Kaggle Model adapter or download final_adapter from HF."""
    if ADAPTER_INPUT is not None:
        adapter = Path(ADAPTER_INPUT)
        if not (adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(f"Missing SFT adapter_config.json: {adapter}")
        return adapter

    candidates = sorted(Path("/kaggle/input").rglob("final_adapter/adapter_config.json"))
    candidates = [p.parent for p in candidates if (p.parent / "adapter_model.safetensors").is_file()]
    if len(candidates) > 1:
        raise ValueError(f"Multiple final_adapter folders found; set ADAPTER_INPUT: {candidates}")
    if candidates:
        return candidates[0]

    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception as exc:
            raise RuntimeError(
                "Add Kaggle Secret HF_TOKEN with read access to the SFT repo, "
                "or attach the final_adapter as a Kaggle Model Input."
            ) from exc
    local = Path("/kaggle/working/air_sft_adapter") / SFT_ADAPTER_SUBDIR
    local.mkdir(parents=True, exist_ok=True)
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        hf_hub_download(
            repo_id=SFT_REPO,
            filename=f"{SFT_ADAPTER_SUBDIR}/{filename}",
            local_dir=str(local.parent),
            token=token,
        )
    return local


def write_jsonl(path, row):
    """Append one JSON object and flush so Kaggle interruptions lose little work."""
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()


def run():
    if not common.torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU before running this script.")
    if LIMIT < 0 or START_INDEX < 0 or STOP_INDEX < 0 or MAX_NEW_TOKENS < 1:
        raise ValueError("LIMIT/START_INDEX/STOP_INDEX must be >= 0 and MAX_NEW_TOKENS > 0")

    # Configure the shared tested functions without passing folder names to the model.
    data_root = common.resolve_data_root(REALTEST_ROOT)
    common.MODEL_ID = MODEL_ID
    common.ADAPTER = resolve_adapter()
    common.MAX_NEW_TOKENS = MAX_NEW_TOKENS
    common.LOAD_IN_4BIT = LOAD_IN_4BIT
    common.ATTENTION_IMPLEMENTATION = ATTENTION_IMPLEMENTATION
    common.REALTEST_ROOT = data_root

    paths, labels = common.realtest_items(data_root)
    total = min(len(paths), LIMIT) if LIMIT else len(paths)
    end = min(total, STOP_INDEX) if STOP_INDEX else total
    if START_INDEX >= end:
        raise ValueError(f"Empty range: START_INDEX={START_INDEX}, end={end}, total={len(paths)}")
    selected = paths[START_INDEX:end]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    completed = set()
    if OUTPUT.is_file():
        with OUTPUT.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {OUTPUT}:{line_number}") from exc
                if row.get("file_name"):
                    completed.add(row["file_name"])
    remaining = [p for p in selected if common.image_key(p, data_root) not in completed]
    print(f"Dataset={data_root} | selected={len(selected)} | "
          f"saved={len(selected) - len(remaining)} | remaining={len(remaining)}", flush=True)
    if not remaining:
        print(f"Nothing to infer. JSONL is complete: {OUTPUT}", flush=True)
        return

    processor = common.load_processor()
    model = common.load_model_with_offload()
    correct = 0
    for index, path in enumerate(remaining, 1):
        key = common.image_key(path, data_root)
        print(f"[{index}/{len(remaining)}] {key}", flush=True)
        prediction, raw, tokens, generation_seconds, elapsed = common.infer_one(path, processor, model)
        gt = labels[key]
        is_correct = prediction.strip().casefold() == gt.strip().casefold()
        row = {
            "file_name": key,
            "ground_truth": gt,
            "prediction": prediction,
            "correct": is_correct,
            "generated_tokens": tokens,
            "generation_seconds": round(generation_seconds, 3),
            "elapsed_seconds": round(elapsed, 3),
            "raw_output": raw,
        }
        write_jsonl(OUTPUT, row)
        correct += int(is_correct)
        print(f"  prediction={prediction!r} | gt={gt!r} | correct={is_correct}", flush=True)

    print(f"Saved {OUTPUT}; current-run accuracy={correct}/{len(remaining)}="
          f"{correct / len(remaining):.2%}", flush=True)


if __name__ == "__main__":
    run()