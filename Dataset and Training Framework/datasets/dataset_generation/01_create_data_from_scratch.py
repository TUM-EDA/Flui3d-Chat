import json
import random
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Type

# Import project-specific modules and classes
from config import dataset_generation_config
from design_sampling import GraphGenerator, JsonConverter
from prompt_generation import PromptGenerator, ProcessOrientedPromptGenerator, ConnectionOrientedPromptGenerator, PathOrientedPromptGenerator

def setup_directories():
    """Create all necessary output directories if they don't exist."""
    print("Setting up output directories...")
    dataset_generation_config.FINAL_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    dataset_generation_config.TEMP_JSON_DIR.mkdir(parents=True, exist_ok=True)
    for style in ["process_style", "connection_style", "assay_style/part_1", "assay_style/part_2"]:
        (dataset_generation_config.OPENAI_INPUT_DIR / style).mkdir(parents=True, exist_ok=True)

def _write_batch_entries(file_path: Path, entries: List[Dict]):
    """Helper to write a list of batch entries to a .jsonl file."""
    with file_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def create_openai_batches(style: str, prompts: List[Dict]):
    """
    Generates and saves OpenAI batch files from a list of prompts.
    """
    print(f"Creating OpenAI batches for '{style}' style...")
    output_dir = dataset_generation_config.OPENAI_INPUT_DIR / f"{style}_style"
    batch = []

    if style in ["process", "connection"]:
        elements_per_entry = 10
        total = len(prompts)
        split_index = int(total * 0.8)

        for i in range(0, total, elements_per_entry):
            entry_data = prompts[i:i + elements_per_entry]
            if not entry_data:
                continue

            # Last 20% of batches use bullet points
            is_bp_batch = i >= split_index
            system_message = dataset_generation_config.SYSTEM_MESSAGES[style]['bp'] if is_bp_batch else dataset_generation_config.SYSTEM_MESSAGES[style]['base']
            response_format = {"type": "json_schema", "json_schema": dataset_generation_config.OPENAI_JSON_SCHEMA}

            user_message = json.dumps([{"prompt": obj["prompt"]} for obj in entry_data], ensure_ascii=False)
            
            batch.append({
                "custom_id": str(entry_data[0]["id"]),
                "method": "POST", "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "system", "content": system_message}, {"role": "user", "content": user_message}],
                    "response_format": response_format,
                    "presence_penalty": random.choice([0.0, 1.0])
                }
            })
        _write_batch_entries(output_dir / f"{style}_descriptions_batch.jsonl", batch)

    elif style == "assay":
        system_message_template = dataset_generation_config.SYSTEM_MESSAGES[style]['base']
        for entry_data in prompts:
            random_number = random.randint(1, 10)
            ordinal = {1: 'st', 2: 'nd', 3: 'rd'}.get(random_number % 10, 'th')
            system_message = system_message_template.replace("{random number}", f"{random_number}{ordinal}")
            user_message = json.dumps([{"prompt": entry_data["prompt"]}], ensure_ascii=False)

            batch.append({
                "custom_id": str(entry_data["id"]),
                "method": "POST", "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "system", "content": system_message}, {"role": "user", "content": user_message}],
                    "response_format": {"type": "text"},
                    "presence_penalty": random.choice([0.0, 1.0])
                }
            })
        
        # Split into 2 files due to OpenAI limitations
        split_index = len(batch) // 2
        _write_batch_entries(output_dir / "part_1" / "path_descriptions_batch_part1.jsonl", batch[:split_index])
        _write_batch_entries(output_dir / "part_2" / "path_descriptions_batch_part2.jsonl", batch[split_index:])
        
    print(f"Successfully created batch files for '{style}'.")


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

        # 3. Create OpenAI batch files from the 'for_llm' prompts
        create_openai_batches(style, prompts_for_llm)

        # 4. Convert graphs to JSON for the GRPO dataset
        print(f"[{style}] Converting graphs to JSON format...")
        json_converter = JsonConverter()
        json_designs = json_converter.convert_graphs(designs_with_ids)

        # 4a. Save JSON designs for the next script (02_extract...)
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
    print("1. Use the created batch files in './output/openai_inputs/' with the OpenAI API.")
    print("2. Download the output files from the OpenAI API batch jobs.")
    print("3. Place the API '.jsonl' output files into their corresponding style folders inside './output/openai_outputs/'.")
    print("   For example, the output for the process-oriented style should go into './output/openai_outputs/process_style/'.")
    print("4. Run the '02_extract_enhanced_prompts.py' script. It will combine the API outputs with the corresponding")
    print("   JSON designs saved in './output/json_designs/' to create the final baseline finetuning dataset.")


if __name__ == "__main__":
    main()