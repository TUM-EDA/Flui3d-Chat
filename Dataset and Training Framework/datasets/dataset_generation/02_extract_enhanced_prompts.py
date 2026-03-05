import json
import random
import re
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

# Import settings from the configuration file
from config import dataset_generation_config

def find_batch_file(directory: Path) -> Optional[Path]:
    """Finds the first .jsonl file in a given directory."""
    if not directory.is_dir():
        print(f"Warning: Directory not found: {directory}")
        return None
    try:
        return next(directory.glob("*.jsonl"))
    except StopIteration:
        print(f"Warning: No '.jsonl' batch file found in {directory}")
        return None

def load_jsonl(file_path: Path) -> List[Dict]:
    """Loads a JSONL file into a list of dictionaries."""
    with file_path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def load_json(file_path: Path) -> List[Dict]:
    """Loads a standard JSON file."""
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)

def extract_json_from_text(text: str) -> Optional[Dict]:
    """Extracts a JSON object from a markdown code block in plain text."""
    # Find the last JSON code block to be robust
    matches = re.findall(r"```json(.*?)```", text, re.DOTALL)
    if not matches:
        return None
        
    json_block = matches[-1]
    # Clean up potential trailing commas that make JSON invalid
    json_block = re.sub(r"[;,]\s*([\]}])", r"\1", json_block)
    try:
        return json.loads(json_block)
    except json.JSONDecodeError:
        print(f"Info: Could not decode extracted JSON block.")
        return None

def process_style_batch(batch_dir: Path, config_entry: Dict) -> List[List[Dict]]:
    """
    Processes a single style's batch output to create conversation triplets.
    
    Args:
        batch_dir: The directory containing the OpenAI batch output file.
        config_entry: The configuration dictionary for this style.

    Returns:
        A list of conversation triplets.
    """
    batch_file = find_batch_file(batch_dir)
    if not batch_file:
        return []

    print(f"Processing batch file: {batch_file.name}...")
    
    # Load the OpenAI output and the corresponding ground-truth designs
    batch_data = load_jsonl(batch_file)
    ground_truth_data = load_json(config_entry["ground_truth_file"])
    is_single_prompt = config_entry["is_single_prompt"]
    
    # Create a lookup map for ground-truth JSONs by their ID
    assistant_lookup = {item['id']: item['json'] for item in ground_truth_data}
    
    conversations = []
    for entry in batch_data:
        content = entry.get("response", {}).get("body", {}).get("choices", [{}])[0].get("message", {}).get("content", "")
        custom_id = int(entry.get("custom_id", 0))

        if not content or not custom_id:
            continue

        prompts_list = []
        # Extract prompts based on the style's configuration
        if is_single_prompt:
            prompt_object = extract_json_from_text(content)
            if prompt_object and "prompt" in prompt_object:
                prompts_list.append(prompt_object["prompt"])
        else:
            try:
                prompts_json = json.loads(content)
                prompts_list = [p["prompt"] for p in prompts_json.get("prompts", [])]
                if len(prompts_list) != 10:
                    print(f"Info: Dropped entry for custom_id {custom_id}. Expected 10 prompts, found {len(prompts_list)}.")
                    continue
            except (json.JSONDecodeError, TypeError):
                print(f"Info: Could not parse JSON content for custom_id {custom_id}.")
                continue

        # Generate conversation triplets for each extracted prompt
        for idx, user_message in enumerate(prompts_list):
            conversation_id = custom_id + idx
            assistant_message_json = assistant_lookup.get(conversation_id)
            
            if assistant_message_json:
                assistant_message = json.dumps(assistant_message_json, indent=2)
                conversations.append([
                    {"role": "system", "content": str(dataset_generation_config.BASELINE_SYSTEM_MESSAGE)},
                    {"role": "user", "content": str(user_message)},
                    {"role": "assistant", "content": str(assistant_message)}
                ])
                
    return conversations

def main():
    """Main function to orchestrate the extraction and combination of data."""
    print("--- Starting Baseline Finetuning Dataset Creation ---")
    all_conversations = []

    # Process each style defined in the configuration
    for style_path, style_config in dataset_generation_config.STYLE_CONFIG.items():
        batch_directory = dataset_generation_config.OPENAI_OUTPUT_DIR / style_path
        print(f"\nProcessing style directory: {batch_directory}")
        
        style_conversations = process_style_batch(batch_directory, style_config)
        all_conversations.extend(style_conversations)
        print(f"Found {len(style_conversations)} conversations for this style.")

    if not all_conversations:
        print("\nNo conversations were generated. Exiting.")
        return

    print(f"\nTotal conversations combined: {len(all_conversations)}")

    # Shuffle the entire dataset for better training distribution
    random.shuffle(all_conversations)
    print("Shuffling complete.")

    # Convert to a pandas DataFrame and save
    df = pd.DataFrame({"conversations": all_conversations})
    
    # Ensure the output directory exists
    dataset_generation_config.FINAL_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    df.to_pickle(dataset_generation_config.BASELINE_SFT_DATASET_PATH)

    print("\n--- Baseline Finetuning Dataset Successfully Created! ---")
    print(f"Saved to: {dataset_generation_config.BASELINE_SFT_DATASET_PATH.resolve()}")

if __name__ == "__main__":
    main()