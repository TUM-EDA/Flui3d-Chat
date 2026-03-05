import argparse
from pathlib import Path
import torch
from unsloth import FastLanguageModel, FastModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only, standardize_data_formats
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from pandas import read_pickle

def load_dataset(dataset_path: Path, num_examples: int = 8192) -> Dataset:
    """
    Loads the dataset from a pickle file and prepares it for training.
    
    Args:
        dataset_path (Path): The path to the .pkl dataset file.
        num_examples (int): The number of examples to select from the dataset.
    
    Returns:
        Dataset: The prepared Hugging Face Dataset object.
    """
    print(f"Loading dataset from {dataset_path.resolve()}...")
    df = read_pickle(dataset_path)
    dataset = Dataset.from_pandas(df)
    
    print(f"Selecting the first {num_examples} examples...")
    dataset = dataset.select(range(min(len(dataset), num_examples)))
    
    dataset = standardize_data_formats(dataset)
    return dataset

def formatting_prompts_func(examples, tokenizer):
    """
    Applies the chat template to a batch of conversations, returning formatted strings.
    """
    return {
        "text": tokenizer.apply_chat_template(
            examples["conversations"],
            tokenize=False,
            add_generation_prompt=False,
        )
    }

# --- Model Specific Configurations ---

# This dictionary holds all the unique parameters for each model.
# If you have already downloaded the base models locally, 
# you can set `model_id` to the local path instead of the Hugging Face ID.
# Simply replace the value of `model_id` with the path to your downloaded model directory.
MODEL_CONFIGS = {
    "llama3_3": {
        "model_id": "unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
        "unsloth_class": FastLanguageModel,
        "max_seq_length": 16384,
        "load_in_4bit": True,
        "peft_settings": {
            "r": 128,
            "lora_alpha": 128,
            "lora_dropout": 0,
            "bias": "none",
            "use_gradient_checkpointing": "unsloth",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        },
        "sft_args": {
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 32,
        },
        "chat_template": "llama-3.1",
        "response_template": {
            "instruction_part": "<|start_header_id|>user<|end_header_id|>\n\n",
            "response_part": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        },
    },
    "qwen2_5": {
        "model_id": "unsloth/Qwen2.5-72B-Instruct-bnb-4bit",
        "unsloth_class": FastLanguageModel,
        "max_seq_length": 32768,
        "load_in_4bit": True,
        "peft_settings": {
            "r": 256,
            "lora_alpha": 256,
            "lora_dropout": 0,
            "bias": "none",
            "use_gradient_checkpointing": "unsloth",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        },
        "sft_args": {
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 64,
        },
        "chat_template": "qwen-2.5",
        "response_template": {
            "instruction_part": "<|im_start|>user\n",
            "response_part": "<|im_start|>assistant\n",
        },
    },
    "qwen3": {
        "model_id": "unsloth/Qwen3-32B",
        "unsloth_class": FastLanguageModel,
        "max_seq_length": 32768,
        "load_in_4bit": False,
        "peft_settings": {
            "r": 128,
            "lora_alpha": 128,
            "lora_dropout": 0,
            "bias": "none",
            "use_gradient_checkpointing": "unsloth",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        },
        "sft_args": {
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 64,
        },
        "chat_template": "qwen-3",
        "response_template": {
            "instruction_part": "<|im_start|>user\n",
            "response_part": "<|im_start|>assistant\n",
        },
    },
    "gemma3": {
        "model_id": "unsloth/gemma-3-27b-it",
        "unsloth_class": FastModel,
        "max_seq_length": 16384,
        "load_in_4bit": False,
        "peft_settings": {
            "finetune_vision_layers": False,
            "finetune_language_layers": True,
            "finetune_attention_modules": True,
            "finetune_mlp_modules": True,
            "r": 256,
            "lora_alpha": 256,
            "lora_dropout": 0,
            "bias": "none",
        },
        "sft_args": {
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 128,
        },
        "chat_template": "gemma-3",
        "response_template": {
            "instruction_part": "<start_of_turn>user\n",
            "response_part": "<start_of_turn>model\n",
        },
    },
}

def run_training(model_name: str, dataset_path: Path, script_dir: Path):
    """
    Main function to run the fine-tuning process for a selected model.
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model '{model_name}' not recognized. Choose from: {list(MODEL_CONFIGS.keys())}")

    print(f"--- Starting fine-tuning for model: {model_name} ---")
    config = MODEL_CONFIGS[model_name]

    # Define output directories relative to the script's location
    sft_output_dir = script_dir / "baseline_outputs"
    final_lora_dir = script_dir / f"baseline_lora_adapters/{model_name}"

    # 1. Load Model and Tokenizer
    model, tokenizer = config["unsloth_class"].from_pretrained(
        model_name=config["model_id"],
        max_seq_length=config["max_seq_length"],
        dtype=None,  # Auto-detection
        load_in_4bit=config["load_in_4bit"],
    )

    # 2. Add LoRA Adapters
    model = config["unsloth_class"].get_peft_model(model, **config["peft_settings"])

    # 3. Prepare Dataset
    tokenizer = get_chat_template(tokenizer, chat_template=config["chat_template"])
    dataset = load_dataset(dataset_path)
    
    dataset = dataset.map(
        formatting_prompts_func,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
    )

    # 4. Configure SFT Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config["max_seq_length"],
        packing=False,
        args=SFTConfig(
            # Core SFT arguments
            per_device_train_batch_size=config["sft_args"]["per_device_train_batch_size"],
            gradient_accumulation_steps=config["sft_args"]["gradient_accumulation_steps"],
            # Shared arguments across all models
            learning_rate=1e-4,
            num_train_epochs=2,
            warmup_steps=5,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="paged_adamw_32bit",
            weight_decay=0.02,
            lr_scheduler_type="cosine",
            output_dir=str(sft_output_dir),
            report_to="none",
        ),
    )

    # 5. Set up training on responses only
    trainer = train_on_responses_only(trainer, **config["response_template"])

    # 6. Start Training
    print("Starting training...")
    trainer.train()
    print("Training finished!")

    # 7. Save LoRA Adapters
    final_lora_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_lora_dir))
    tokenizer.save_pretrained(str(final_lora_dir))
    print(f"Model adapters saved to {final_lora_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune various language models using Unsloth.")
    parser.add_argument(
        "model_name",
        type=str,
        choices=list(MODEL_CONFIGS.keys()),
        help="The name of the model to fine-tune."
    )
    args = parser.parse_args()
    
    script_dir =  Path(__file__).resolve().parent
    dataset_file_path = script_dir / "../datasets/final_datasets/baseline_sft_dataset.pkl"
    
    run_training(args.model_name, dataset_file_path, script_dir)