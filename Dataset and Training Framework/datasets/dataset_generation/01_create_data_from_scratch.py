import json
import random
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Type

# Import project-specific modules and classes
from config import dataset_generation_config
from design_sampling import GraphGenerator, JsonConverter
from prompt_generation import PromptGenerator, ProcessOrientedPromptGenerator, ConnectionOrientedPromptGenerator, PathOrientedPromptGenerator

STYLES = ["process", "connection", "assay"]


def setup_directories():
    """Create all necessary output directories if they don't exist."""
    print("Setting up output directories...")
    dataset_generation_config.FINAL_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    dataset_generation_config.TEMP_JSON_DIR.mkdir(parents=True, exist_ok=True)
    for style in STYLES:
        (dataset_generation_config.OPENAI_INPUT_DIR / f"{style}_style").mkdir(parents=True, exist_ok=True)


def _write_batch_entries(file_path: Path, entries: List[Dict]):
    """Helper to write a list of batch entries to a .jsonl file."""
    with file_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _weighted_choice(distribution: Dict[str, float]) -> str:
    """Samples a single key from a {value: probability} distribution."""
    return random.choices(list(distribution), weights=list(distribution.values()), k=1)[0]


def _sample_style_directives() -> Dict[str, Any]:
    """Samples one style-directive vector that steers HOW a prompt is phrased.

    Diversity across prompts is driven here (controllable/reportable) rather than left to the
    model's default voice; the generation system message defines what each token means.
    """
    return {
        "register": _weighted_choice(dataset_generation_config.REGISTER_DISTRIBUTION),
        "format": _weighted_choice(dataset_generation_config.FORMAT_DISTRIBUTION),
        "length": _weighted_choice(dataset_generation_config.LENGTH_DISTRIBUTION),
        "intro": random.random() < dataset_generation_config.INTRO_PROBABILITY,
    }


def create_openai_batches(style: str, prompts: List[Dict]):
    """
    Generates and saves an OpenAI batch file for one prompt style.

    One structured description is sent per API call (batch size 1) at gpt-5.4. The user message
    carries the template ``draft`` plus a sampled ``style`` directive vector (and, for the assay
    style, a sampled ``assay_domain``); the (constant, cacheable) system message owns the rewriting
    contract. The response is a single ``{"prompt": ...}`` object.
    """
    print(f"Creating OpenAI batch for '{style}' style...")
    output_dir = dataset_generation_config.OPENAI_INPUT_DIR / f"{style}_style"
    system_message = dataset_generation_config.SYSTEM_MESSAGES[style]
    response_format = {"type": "json_schema", "json_schema": dataset_generation_config.OPENAI_JSON_SCHEMA}

    batch = []
    for entry in prompts:
        payload: Dict[str, Any] = {"draft": entry["prompt"], "style": _sample_style_directives()}
        if style == "assay":
            payload["assay_domain"] = random.choice(dataset_generation_config.ASSAY_DOMAIN_POOL)

        batch.append({
            "custom_id": str(entry["id"]),
            "method": "POST", "url": "/v1/chat/completions",
            "body": {
                "model": dataset_generation_config.OPENAI_GEN_MODEL,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                "response_format": response_format,
                "reasoning_effort": dataset_generation_config.OPENAI_REASONING_EFFORT,
                "verbosity": dataset_generation_config.OPENAI_VERBOSITY,
            },
        })

    _write_batch_entries(output_dir / f"{style}_descriptions_batch.jsonl", batch)
    print(f"Successfully created batch file for '{style}' ({len(batch)} prompts).")


def create_grpo_dataset(all_prompts: List[Dict], all_jsons: List[Dict]):
    """
    Creates the final GRPO dataset from prompts and their corresponding JSON designs.
    """
    print("\nCreating final GRPO dataset...")
    if len(all_prompts) != len(all_jsons):
        raise ValueError("Mismatched number of prompts and JSON designs. Cannot create dataset.")

    formatted_prompts = []
    formatted_jsons = []

    for prompt, json_design in zip(all_prompts, all_jsons):
        # Format the prompt into a system-user conversation
        formatted_prompts.append([
            {"role": "system", "content": dataset_generation_config.GRPO_SYSTEM_MESSAGE},
            {"role": "user", "content": str(prompt["prompt"])}
        ])
        # Format the JSON design as a string
        formatted_jsons.append(json.dumps(json_design, indent=2))

    # Combine into pairs, shuffle, and create DataFrame
    combined_data = list(zip(formatted_prompts, formatted_jsons))
    random.shuffle(combined_data)
    df = pd.DataFrame(combined_data, columns=["prompt", "json_chip_design"])

    # Save to a pickle file for efficient loading later
    df.to_pickle(dataset_generation_config.GRPO_DATASET_PATH)
    print(f"Dataset successfully created with {len(df)} entries.")
    print(f"Saved to: {dataset_generation_config.GRPO_DATASET_PATH.resolve()}")


def main():
    """Main execution script to generate all data from scratch."""
    setup_directories()

    prompt_styles: Dict[str, Type[PromptGenerator]] = {
        "process": ProcessOrientedPromptGenerator,
        "connection": ConnectionOrientedPromptGenerator,
        "assay": PathOrientedPromptGenerator
    }

    all_prompts_for_grpo = []
    all_jsons_for_grpo = []

    for style, GeneratorClass in prompt_styles.items():
        print(f"\n--- Processing Style: {style.upper()} ---")

        # 1. Generate a unique set of graphs for this style
        print(f"[{style}] Generating {dataset_generation_config.MAX_DESIGNS_PER_STYLE} graphs...")
        graph_gen = GraphGenerator(
            max_components=dataset_generation_config.MAX_COMPONENTS,
            max_junctions=dataset_generation_config.MAX_JUNCTIONS,
            max_designs=dataset_generation_config.MAX_DESIGNS_PER_STYLE
        )
        designs_with_ids = graph_gen.generate_designs()

        # 2. Generate prompts using the style-specific generator
        print(f"[{style}] Generating prompts...")
        prompt_gen = GeneratorClass(designs_with_ids)
        prompts_for_llm, prompts_wo_llm = prompt_gen.generate_prompts()

        # 3. Create the OpenAI batch file from the 'for_llm' template drafts
        create_openai_batches(style, prompts_for_llm)

        # 4. Convert graphs to JSON (ground truth for the SFT/GRPO targets)
        print(f"[{style}] Converting graphs to JSON format...")
        json_converter = JsonConverter()
        json_designs = json_converter.convert_graphs(designs_with_ids)

        # 4a. Save JSON designs for the next script (final dataset assembly)
        print(f"[{style}] Saving JSON designs for later use...")
        temp_json_path = dataset_generation_config.TEMP_JSON_DIR / f"{style}_designs.json"
        with temp_json_path.open("w", encoding="utf-8") as f:
            json.dump(json_designs, f, indent=2)

        # 5. Collect data for the final combined GRPO dataset
        id_to_json_map = {item['id']: item['json'] for item in json_designs}
        for prompt in prompts_wo_llm:
            prompt_id = int(prompt['id'])
            if prompt_id in id_to_json_map:
                all_prompts_for_grpo.append(prompt)
                all_jsons_for_grpo.append(id_to_json_map[prompt_id])

    # 6. After processing all styles, create the final dataset
    create_grpo_dataset(all_prompts_for_grpo, all_jsons_for_grpo)

    # --- Final Informational Message ---
    print("\n\n--- All initial tasks completed successfully! ---")
    print("\nNEXT STEPS:")
    print("1. Submit the batch files in './output/openai_inputs/<style>_style/' to the OpenAI Batch API.")
    print("2. Download each batch's output and place the '.jsonl' result into the matching folder")
    print("   under './output/openai_outputs/<style>_style/'.")
    print("3. Run '02_extract_enhanced_prompts.py' to pair the enhanced prompts with their JSON")
    print("   designs and build the baseline finetuning dataset.")


if __name__ == "__main__":
    main()
