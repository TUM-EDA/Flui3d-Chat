import os
import json
from pathlib import Path

# --- General Generation Parameters ---
MAX_COMPONENTS = 20
MAX_JUNCTIONS = 10
MAX_DESIGNS_PER_STYLE = 20000

# --- Directory Configuration ---
# Creates a main output directory to hold all generated files
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_OUTPUT_DIR = SCRIPT_DIR / "../output"
OPENAI_INPUT_DIR = BASE_OUTPUT_DIR / "openai_inputs"
OPENAI_OUTPUT_DIR = BASE_OUTPUT_DIR / "openai_outputs"
TEMP_JSON_DIR = BASE_OUTPUT_DIR / "json_designs"
FINAL_DATASET_DIR = SCRIPT_DIR / "../../final_datasets"
GRPO_DATASET_PATH = FINAL_DATASET_DIR / "grpo_dataset.pkl"
BASELINE_SFT_DATASET_PATH = FINAL_DATASET_DIR / "baseline_sft_dataset.pkl"

# --- OpenAI Batch Configuration ---

def _load_message(filepath):
    """Helper to load message content from a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"System message file not found at {filepath}. Please add the required prompt files.")


RESOURCE_DIR = SCRIPT_DIR / "../../resources"
PROMPT_DIR = RESOURCE_DIR / "system_messages"
JSON_SCHEMA_DIR = RESOURCE_DIR / "json_schemas"
PROMPT_DIR_OPENAI = PROMPT_DIR / "openai"

SYSTEM_MESSAGES = {
    "process": {
        "base": _load_message(PROMPT_DIR_OPENAI / "process_style.txt"),
        "bp": _load_message(PROMPT_DIR_OPENAI / "process_style_bp.txt"),
    },
    "connection": {
        "base": _load_message(PROMPT_DIR_OPENAI / "connection_style.txt"),
        "bp": _load_message(PROMPT_DIR_OPENAI / "connection_style_bp.txt"),
    },
    "assay": {
        "base": _load_message(PROMPT_DIR_OPENAI / "assay_style.txt"),
    }
}

# --- Style Processing Configuration ---
STYLE_CONFIG = {
    "process_style": {
        "ground_truth_file": TEMP_JSON_DIR / "process_designs.json",
        "is_single_prompt": False, # Expects a list of 10 prompts per entry
    },
    "connection_style": {
        "ground_truth_file": TEMP_JSON_DIR / "connection_designs.json",
        "is_single_prompt": False, # Expects a list of 10 prompts per entry
    },
    "assay_style/part_1": {
        "ground_truth_file": TEMP_JSON_DIR / "assay_designs.json",
        "is_single_prompt": True, # Expects one prompt extracted from plain text
    },
    "assay_style/part_2": {
        "ground_truth_file": TEMP_JSON_DIR / "assay_designs.json",
        "is_single_prompt": True, # Expects one prompt extracted from plain text
    },
}


# Load the JSON schema for OpenAI API calls
JSON_SCHEMA_PATH = JSON_SCHEMA_DIR / "openai_output_schema.json"
try:
    with open(JSON_SCHEMA_PATH, 'r', encoding='utf-8') as f:
        OPENAI_JSON_SCHEMA = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    raise FileNotFoundError(f"JSON schema not found or invalid at {JSON_SCHEMA_PATH}. Please add the required schema file.")

# --- GRPO Dataset Configuration ---
# The main system message for the final dataset generation.
GRPO_SYSTEM_MESSAGE = _load_message(PROMPT_DIR / "reasoning.txt")

# --- Baseline SFT Dataset Configuration ---
# The main system message for the final dataset generation.
BASELINE_SYSTEM_MESSAGE = _load_message(PROMPT_DIR / "baseline.txt")