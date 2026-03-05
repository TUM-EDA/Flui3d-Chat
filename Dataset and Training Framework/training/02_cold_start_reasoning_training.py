import argparse
from pathlib import Path
import torch
from unsloth import FastLanguageModel, FastModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only, standardize_data_formats
from trl import SFTTrainer, SFTConfig
from datasets import Dataset, load_from_disk, concatenate_datasets
from pandas import read_pickle

# --- Data Loading and Formatting ---

def load_cold_start_dataset(cold_start_path: Path, final_path: Path) -> Dataset:
    """
    Loads, combines, and prepares the datasets for cold-start fine-tuning.
    """
    print(f"Loading GRPO cold-start dataset from: {cold_start_path.resolve()}")
    try:
        cold_start_dataset = load_from_disk(str(cold_start_path))['train']
    except Exception as e:
        print(f"Error: Could not load cold-start dataset. Please check the path. Details: {e}")
        exit()

    print(f"Loading main dataset from: {final_path.resolve()}")
    try:
        df = read_pickle(final_path)
        final_dataset = Dataset.from_pandas(df)
        final_dataset = final_dataset.rename_column("conversations", "messages")
        # Select a small, random subset
        final_dataset = final_dataset.shuffle().select(range(min(30, len(final_dataset))))
    except Exception as e:
        print(f"Error: Could not load main .pkl dataset. Please check the path. Details: {e}")
        exit()

    print("Combining and shuffling datasets...")
    combined = concatenate_datasets([cold_start_dataset, final_dataset])
    combined = combined.shuffle()
    
    combined = standardize_data_formats(combined)
    return combined

def formatting_prompts_func(examples, tokenizer):
    """
    Applies the chat template to a batch of conversations, returning formatted strings.
    """
    return {
        "text": tokenizer.apply_chat_template(
            examples["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
    }

# --- Model Specific Configurations ---

MODEL_CONFIGS = {
    "llama3_3": {
        "model_id": "./baseline_lora_adapters/llama3_3",
        "unsloth_class": FastLanguageModel,
        "max_seq_length": 16384,
        "load_in_4bit": True,
        "chat_template": "llama-3.1",
        "response_template": {"instruction_part": "<|start_header_id|>user<|end_header_id|>\n\n", "response_part": "<|start_header_id|>assistant<|end_header_id|>\n\n"},
    },
    "qwen2_5": {
        "model_id": "./baseline_lora_adapters/qwen2_5",
        "unsloth_class": FastLanguageModel,
        "max_seq_length": 32768,
        "load_in_4bit": True,
        "chat_template": "qwen-2.5",
        "response_template": {"instruction_part": "<|im_start|>user\n", "response_part": "<|im_start|>assistant\n"},
    },
    "qwen3": {
        "model_id": "./baseline_lora_adapters/qwen3",
        "unsloth_class": FastLanguageModel,
        "max_seq_length": 32768,
        "load_in_4bit": False,
        "chat_template": "qwen-3",
        "response_template": {"instruction_part": "<|im_start|>user\n", "response_part": "<|im_start|>assistant\n"},
    },
    "gemma3": {
        "model_id": "./baseline_lora_adapters/gemma3",
        "unsloth_class": FastModel,
        "max_seq_length": 16384,
        "load_in_4bit": False,
        "chat_template": "gemma-3",
        "response_template": {"instruction_part": "<start_of_turn>user\n", "response_part": "<start_of_turn>model\n"},
    },
}

def run_training(model_name: str, cold_start_dataset_path: Path, final_dataset_path: Path, script_dir: Path):
    """
    Main function to run the cold-start fine-tuning process.
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model '{model_name}' not recognized. Choose from: {list(MODEL_CONFIGS.keys())}")

    print(f"--- Starting COLD-START fine-tuning for model: {model_name} ---")
    config = MODEL_CONFIGS[model_name]

    # Define directories relative to the script's location
    model_path = script_dir / config["model_id"]
    sft_output_dir = script_dir / "cold_start_reasoning_outputs"
    final_lora_dir = script_dir / f"cold_start_reasoning_lora_adapters/{model_name}"

    # 1. Load Model and Tokenizer from previously saved adapters
    # Unsloth automatically handles loading the base model and applying the adapter
    model, tokenizer = config["unsloth_class"].from_pretrained(
        model_name=str(model_path),
        max_seq_length=config["max_seq_length"],
        dtype=None,
        load_in_4bit=config["load_in_4bit"],
    )
    # NOTE: We DO NOT call get_peft_model again, as we are continuing to train the loaded adapters.

    # 2. Prepare Dataset using the new loading function
    tokenizer = get_chat_template(tokenizer, chat_template=config["chat_template"])
    dataset = load_cold_start_dataset(cold_start_dataset_path, final_dataset_path)
    
    dataset = dataset.map(
        formatting_prompts_func,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
    )

    # 3. Configure SFT Trainer with the new parameters
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config["max_seq_length"],
        packing=False,
        args=SFTConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=0,
            num_train_epochs=2,
            learning_rate=2e-4,
            weight_decay=0.02,
            lr_scheduler_type="cosine",
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="paged_adamw_32bit",
            output_dir=str(sft_output_dir),
            report_to="none",
        ),
    )

    # 4. Set up training on responses only
    trainer = train_on_responses_only(trainer, **config["response_template"])

    # 5. Start Training
    print("Starting training...")
    trainer.train()
    print("Training finished!")

    # 6. Save final LoRA Adapters
    final_lora_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_lora_dir))
    tokenizer.save_pretrained(str(final_lora_dir))
    print(f"Model adapters saved to {final_lora_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run cold-start fine-tuning for various language models using Unsloth.")
    parser.add_argument(
        "model_name",
        type=str,
        choices=list(MODEL_CONFIGS.keys()),
        help="The name of the model to fine-tune."
    )
    args = parser.parse_args()
    
    # Define paths for the two dataset components
    script_dir =  Path(__file__).resolve().parent
    cold_start_dataset_path = script_dir / "../datasets/final_datasets/cold_start_reasoning_sft_dataset"
    final_dataset_path = script_dir / "../datasets/final_datasets/baseline_sft_dataset.pkl"
    
    run_training(args.model_name, cold_start_dataset_path, final_dataset_path, script_dir)