# --------------------------------------------------------------------------------------
# IMPORTANT NOTE: UNTESTED SCRIPT
# --------------------------------------------------------------------------------------
# This script and its training parameters are experimental and have not been
# fully tested. The GRPO training process is resource-intensive. At the time of
# development, Unsloth's support for multi-GPU training was limited, and the
# memory requirements exceeded what was available on a single GPU.
#
# Please review and adjust parameters carefully based on your hardware.
# --------------------------------------------------------------------------------------

import argparse
import re
from pathlib import Path
import numpy as np
import torch
from unsloth import FastLanguageModel, FastModel, PatchFastRL
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset
from pandas import read_pickle
from vllm import SamplingParams

from grpo_reward_model import MicrofluidicDesignComparator

# --- Data and Formatting Functions ---

def load_dataset(dataset_path: Path, num_examples: int = 1000) -> Dataset:
    """Loads the GRPO dataset from a pickle file."""
    print(f"Loading dataset from: {dataset_path}")
    df = read_pickle(dataset_path)
    dataset = Dataset.from_pandas(df)
    dataset = dataset.shuffle()
    
    print(f"Selecting {num_examples} examples for GRPO training...")
    return dataset.select(range(min(len(dataset), num_examples)))

def calculate_lengths(dataset: Dataset, tokenizer) -> tuple[int, int]:
    """Calculates max prompt and completion lengths based on the dataset."""
    print("Calculating sequence lengths...")
    tokenized = dataset.map(
        lambda x: {"tokens": tokenizer.apply_chat_template(x["prompt"], add_generation_prompt=True, tokenize=True)},
        batched=True,
    )
    tokenized = tokenized.map(lambda x: {"L": len(x["tokens"])})
    
    # Use the full dataset quantile for a robust max length
    max_prompt_len = int(np.quantile(tokenized["L"], 1)) + 1  # +1 for safety
    max_completion_len = 16384 - max_prompt_len
    
    print(f"Calculated Max Prompt Length: {max_prompt_len}")
    print(f"Calculated Max Completion Length: {max_completion_len}")
    return max_prompt_len, max_completion_len

def extract_xml_json_chip_design(text: str) -> str:
    """Extracts the JSON content from the model's full completion."""
    # Splits the string at the end of the <think> block
    answer = text.split("</think>\n\n")[-1]
    return answer.strip()


# --- Model Specific Configurations ---

MODEL_CONFIGS = {
    "llama3_3": {
        "model_id": "./cold_start_reasoning_lora_adapters/llama3_3",
        "unsloth_class": FastLanguageModel,
        "max_seq_length": 16384,
        "load_in_4bit": True,
        "lora_rank": 128,
    },
    "qwen2_5": {
        "model_id": "./cold_start_reasoning_lora_adapters/qwen2_5",
        "unsloth_class": FastLanguageModel,
        "max_seq_length": 32768,
        "load_in_4bit": True,
        "lora_rank": 256,
    },
    "qwen3": {
        "model_id": "./cold_start_reasoning_lora_adapters/qwen3",
        "unsloth_class": FastLanguageModel,
        "max_seq_length": 32768,
        "load_in_4bit": False,
        "lora_rank": 128,
    },
    "gemma3": {
        "model_id": "./cold_start_reasoning_lora_adapters/gemma3",
        "unsloth_class": FastModel,
        "max_seq_length": 16384,
        "load_in_4bit": False,
        "lora_rank": 256,
    },
}

def run_grpo_training(model_name: str, dataset_path: Path, script_dir: Path):
    """
    Main function to run the GRPO fine-tuning process.
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model '{model_name}' not recognized. Choose from: {list(MODEL_CONFIGS.keys())}")

    print(f"--- Starting GRPO fine-tuning for model: {model_name} ---")
    config = MODEL_CONFIGS[model_name]

    PatchFastRL("GRPO", config["unsloth_class"])

    # --- Setup ---
    model_path = script_dir / config["model_id"]
    output_dir = script_dir / f"grpo_reasoning_lora_adapters/{model_name}"
    training_output_dir = script_dir / "grpo_reasoning_outputs"
    json_schema_path = script_dir / "../datasets/resources/json_schemas/microfluidic_schema.json"
    
    # 1. Load Model and Tokenizer from previously saved adapters
    # Unsloth automatically handles loading the base model and applying the adapter
    model, tokenizer = config["unsloth_class"].from_pretrained(
        model_name=str(model_path),
        max_seq_length=config["max_seq_length"],
        dtype=None,
        load_in_4bit=config["load_in_4bit"],
        max_lora_rank=config["lora_rank"],
        fast_inference=True,
        gpu_memory_utilization=0.9, # Reduce if out of memory
    )
    # NOTE: We DO NOT call get_peft_model again, as we are continuing to train the loaded adapters.

    # 2. Load and Prepare Dataset
    dataset = load_dataset(dataset_path)
    max_prompt_length, max_completion_length = calculate_lengths(dataset, tokenizer)
    
    # 3. Define Reward Functions
    design_comparator = MicrofluidicDesignComparator(json_schema_path)

    def microfluidic_reward_func(prompts, completions, json_chip_design, **kwargs) -> list[float]:
        responses = [comp[0]['content'] for comp in completions]
        q = prompts[0][-1]['content']
        extracted_responses = [extract_xml_json_chip_design(r) for r in responses]
        # print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{json_chip_design[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
        # Evaluate the generated design against the ground-truth design
        return [design_comparator.evaluate_design(ground_truth, generated) for ground_truth, generated in zip(json_chip_design, extracted_responses)]

    def strict_format_reward_func(completions, **kwargs) -> list[float]:
        """Reward for strictly adhering to the <think>...</think>\n\n{JSON} format."""
        pattern = r"^<think>\n([\s\S]*?)\n</think>\n\n([\s\S]*?)\n$"
        responses = [comp[0]["content"] for comp in completions]
        matches = [re.match(pattern, r.strip()) for r in responses]
        return [2.0 if match else 0.0 for match in matches]

    # 4. Configure GRPOTrainer
    
    vllm_sampling_params = SamplingParams(
        min_p = 0.9,
        stop = [tokenizer.eos_token],
        include_stop_str_in_output = True,
    )

    training_args = GRPOConfig(
        use_vllm=True,
        learning_rate=5e-6,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        num_generations=8,
        num_train_epochs=1,
        # max_steps=50,
        # save_steps=50,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_32bit",
        logging_steps=1,
        report_to="none",
        output_dir=str(training_output_dir),
        # GRPO-specific generation params
        temperature=0.3,
        vllm_sampling_params = vllm_sampling_params,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            strict_format_reward_func,
            microfluidic_reward_func
        ],
        args=training_args,
        train_dataset=dataset,
    )
    
    # 5. Start Training
    print("Starting GRPO training...")
    trainer.train()
    print("Training finished!")

    # 6. Save final LoRA adapters
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"GRPO-trained model adapters saved to {output_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GRPO fine-tuning for various language models.")
    parser.add_argument(
        "model_name",
        type=str,
        choices=list(MODEL_CONFIGS.keys()),
        help="The name of the model to fine-tune."
    )
    args = parser.parse_args()
    
    script_directory = Path(__file__).resolve().parent
    dataset_file_path = script_directory / "../datasets/final_datasets/grpo_dataset.pkl"
    
    run_grpo_training(args.model_name, dataset_file_path, script_directory)