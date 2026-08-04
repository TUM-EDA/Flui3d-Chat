import os
import json
from pathlib import Path

# --- General Generation Parameters ---
MAX_COMPONENTS = 20
MAX_JUNCTIONS = 10
# 3 styles x 3334 -> 10,002 unique graphs total (Decision #12).
MAX_DESIGNS_PER_STYLE = 3334

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

# --- Aux-LLM generation pass (OpenAI Batch API) ---
# One natural-language prompt is generated per structured description (batch size 1), so the
# bullet-point variant is now a value of the `format` style directive rather than a separate
# system message. Each style therefore has exactly one (constant, cacheable) system message.
OPENAI_GEN_MODEL = "gpt-5.4"
OPENAI_BATCH_SIZE = 1
# GPT-5-family decoding controls, sent per request in the batch body.
# `medium` reasoning gives reliable instruction-following on the rewriting contract without
# over-thinking a naturalization task; output length is already governed by the `length`
# style directive, so `medium` verbosity keeps the model from padding or clipping the prompt.
OPENAI_REASONING_EFFORT = "medium"
OPENAI_VERBOSITY = "medium"

SYSTEM_MESSAGES = {
    "process": _load_message(PROMPT_DIR_OPENAI / "process_style.txt"),
    "connection": _load_message(PROMPT_DIR_OPENAI / "connection_style.txt"),
    "assay": _load_message(PROMPT_DIR_OPENAI / "assay_style.txt"),
}

# --- Diversity: style-directive vector ---
# One directive vector is sampled per prompt and embedded (as bare token keys) into the user
# message. The generation system message defines what each token means, so the sampled
# distribution stays controllable/reportable here while the semantics live (and cache) in the
# system message. Token names MUST match the system messages.
REGISTER_DISTRIBUTION = {
    "expert": 0.30,       # precise microfluidics domain expert
    "engineer": 0.20,     # terse, telegraphic lab/process engineer
    "student": 0.20,      # verbose graduate student, explains intent
    "beginner": 0.15,     # casual newcomer, plain words
    "non_native": 0.15,   # simple grammar, occasionally awkward but clear
}
FORMAT_DISTRIBUTION = {
    "prose": 0.50,        # flowing paragraph(s)
    "bullets": 0.20,      # bulleted list (absorbs the old bullet-point system message)
    "numbered": 0.15,     # numbered steps
    "run_on": 0.10,       # single dense run-on request
    "qa": 0.05,           # phrased as a question to the assistant
}
LENGTH_DISTRIBUTION = {
    "short": 0.30,
    "medium": 0.50,
    "long": 0.20,
}
# Probability that the prompt opens with a conversational intro/greeting rather than diving in.
INTRO_PROBABILITY = 0.6

# Assay/path style only: a domain is sampled per prompt to spread real-assay coverage instead of
# letting the model default to the same few assays. Replaces the old "pick the Nth assay" trick.
# The model still invents the concrete assay name/materials; the domain just steers the area.
ASSAY_DOMAIN_POOL = [
    # --- Sample prep & separation ---
    "blood plasma / cell separation",
    "circulating tumor cell (CTC) isolation",
    "exosome / extracellular vesicle isolation",
    "nucleic acid extraction & purification",
    "white blood cell enrichment / leukapheresis",
    "platelet isolation",
    "size-based bioparticle fractionation",
    "sperm sorting & motility analysis",
    # --- Genomics & single-cell ---
    "droplet digital PCR",
    "single-cell RNA sequencing preparation",
    "single-cell encapsulation in hydrogel beads",
    "whole-genome amplification",
    "NGS library preparation",
    "CRISPR screening",
    "digital nucleic acid quantification",
    # --- Immunoassays & protein analysis ---
    "on-chip immunoassay (ELISA)",
    "digital (single-molecule) immunoassay",
    "multiplexed bead-based cytokine profiling",
    "antibody discovery / hybridoma screening",
    "on-chip protein crystallization screening",
    "enzyme kinetics assay",
    "aptamer selection (SELEX)",
    "on-chip proteomic digestion",
    # --- Organ-on-chip & cell biology ---
    "organ-on-chip perfusion",
    "liver-on-chip hepatotoxicity testing",
    "lung-on-chip model",
    "gut / intestine-on-chip model",
    "kidney-on-chip model",
    "blood-brain-barrier model",
    "heart-on-chip / cardiomyocyte assay",
    "tumor-on-chip drug testing",
    "spheroid / organoid formation",
    "pancreatic islet perfusion (insulin secretion)",
    "neuron-on-chip / neural culture",
    "co-culture interaction studies",
    "chemical gradient generation for chemotaxis",
    "wound-healing / cell migration assay",
    "cell electroporation / transfection",
    "impedance-based cell counting",
    # --- Drug & particle synthesis ---
    "drug dose-response screening",
    "drug solubility / formulation screening",
    "nanoparticle / liposome synthesis",
    "mRNA-lipid-nanoparticle vaccine formulation",
    "siRNA delivery particle synthesis",
    "controlled drug-release microparticle synthesis",
    "microgel / Janus particle synthesis",
    "double-emulsion templating",
    "quantum dot synthesis",
    # --- Diagnostics & point-of-care ---
    "point-of-care diagnostic panel",
    "viral load quantification",
    "respiratory virus (e.g. SARS-CoV-2) detection",
    "malaria parasite detection",
    "sepsis biomarker panel",
    "cardiac troponin point-of-care test",
    "blood coagulation / clotting-time assay",
    "glucose / lactate metabolite monitoring",
    # --- Microbiology & environmental ---
    "bacterial culture & antibiotic susceptibility testing",
    "antibiotic minimum inhibitory concentration (MIC) assay",
    "phage therapy screening",
    "microbiome co-culture & analysis",
    "emulsion-based directed evolution",
    "synthetic-biology strain screening",
    "water-quality / environmental pathogen detection",
    "food-safety pathogen detection",
    # --- Chemistry & physics ---
    "continuous-flow chemical synthesis",
    "catalyst reaction screening",
    "electrochemical sensing",
    "metabolomics sample preparation",
]

# --- Style processing configuration ---
# Keyed by short style name; output/input subdirectories are derived as f"{style}_style".
STYLE_CONFIG = {
    "process": {"ground_truth_file": TEMP_JSON_DIR / "process_designs.json"},
    "connection": {"ground_truth_file": TEMP_JSON_DIR / "connection_designs.json"},
    "assay": {"ground_truth_file": TEMP_JSON_DIR / "assay_designs.json"},
}


# Load the JSON schema for OpenAI API calls (single-prompt response format).
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
