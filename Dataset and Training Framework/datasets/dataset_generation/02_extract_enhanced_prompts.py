"""Assemble the baseline SFT dataset from the aux-LLM-enhanced prompts.

Final stage of the data pipeline. Reads the enhanced prompts returned by the OpenAI Batch API
(``output/openai_outputs/<style>_style/*.jsonl``), pairs each with its ground-truth JSON design by
``custom_id``, and builds 3-role conversations (system = baseline.txt, user = enhanced prompt,
assistant = JSON design). The combined, shuffled set is saved as
``final_datasets/baseline_sft_dataset.pkl``.

Each request produced exactly one prompt (batch size 1), returned as a strict-schema JSON object
``{"prompt": ...}`` (see ``resources/json_schemas/openai_output_schema.json``).

Run from datasets/dataset_generation/ after downloading the batch outputs.
"""

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from config import dataset_generation_config as cfg


def _find_batch_file(directory: Path) -> Optional[Path]:
    """Returns the single .jsonl batch-output file in a style directory (or None)."""
    if not directory.is_dir():
        print(f"  Warning: directory not found: {directory}")
        return None
    files = sorted(directory.glob("*.jsonl"))
    if not files:
        print(f"  Warning: no .jsonl output file in {directory}")
        return None
    return files[0]


def _load_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_ground_truth(path: Path) -> Dict[str, Dict]:
    """Loads ground-truth designs keyed by stringified id."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(item["id"]): item["json"] for item in data}


def _extract_prompt(content: str) -> Optional[str]:
    """Pulls the ``prompt`` string from the aux-LLM response content.

    Responses use structured output, so ``content`` is normally raw JSON; markdown fences and
    surrounding prose are tolerated as a fallback.
    """
    if not content:
        return None
    obj = None
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        fenced = re.findall(r"```(?:json)?(.*?)```", content, re.DOTALL)
        blocks = fenced if fenced else re.findall(r"(\{.*\})", content, re.DOTALL)
        for block in reversed(blocks):
            cleaned = re.sub(r"[;,]\s*([\]}])", r"\1", block.strip())
            try:
                obj = json.loads(cleaned)
                break
            except json.JSONDecodeError:
                continue
    if isinstance(obj, dict) and isinstance(obj.get("prompt"), str) and obj["prompt"].strip():
        return obj["prompt"].strip()
    return None


def build_conversations() -> List[List[Dict]]:
    """Pairs each enhanced prompt with its JSON design into a 3-role conversation."""
    conversations: List[List[Dict]] = []

    for style in cfg.STYLE_CONFIG:
        out_dir = cfg.OPENAI_OUTPUT_DIR / f"{style}_style"
        ground_truth_path = cfg.STYLE_CONFIG[style]["ground_truth_file"]

        batch_file = _find_batch_file(out_dir)
        if batch_file is None:
            continue
        if not ground_truth_path.is_file():
            print(f"  Warning: no ground-truth file for '{style}' at {ground_truth_path}; skipping.")
            continue

        ground_truth = _load_ground_truth(ground_truth_path)

        matched = 0
        dropped = 0
        for entry in _load_jsonl(batch_file):
            cid = str(entry.get("custom_id", "")).strip()
            content = (entry.get("response", {}).get("body", {})
                       .get("choices", [{}])[0].get("message", {}).get("content", ""))
            prompt = _extract_prompt(content)
            design = ground_truth.get(cid)
            if prompt is None or design is None:
                dropped += 1
                continue
            conversations.append([
                {"role": "system", "content": str(cfg.BASELINE_SYSTEM_MESSAGE)},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": json.dumps(design, indent=2)},
            ])
            matched += 1
        print(f"[{style}] {matched} conversations (dropped {dropped}) from {batch_file.name}.")

    return conversations


def main():
    print("--- Building baseline finetuning dataset ---")
    conversations = build_conversations()

    if not conversations:
        print("\nNo conversations were generated. Exiting.")
        return

    random.shuffle(conversations)
    print(f"\nTotal conversations: {len(conversations)} (shuffled).")

    df = pd.DataFrame({"conversations": conversations})
    cfg.FINAL_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cfg.BASELINE_SFT_DATASET_PATH)

    print("\n--- Baseline finetuning dataset created! ---")
    print(f"Saved to: {cfg.BASELINE_SFT_DATASET_PATH.resolve()}")


if __name__ == "__main__":
    main()
