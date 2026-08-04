import argparse
from pathlib import Path
import torch
from unsloth import FastLanguageModel, FastModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only, standardize_data_formats
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback
from datasets import Dataset
from pandas import read_pickle

# --- Training constants ---
EVAL_SET_SIZE = 512            # examples held out from the dataset for eval / early stopping
EVAL_STEPS = 50                # evaluate + checkpoint every N optimizer steps
EARLY_STOPPING_PATIENCE = 3    # stop after this many evals with no eval_loss improvement


def load_dataset(dataset_path: Path, num_examples: int | None = None) -> Dataset:
    """
    Loads the dataset from a pickle file and prepares it for training.

    Args:
        dataset_path (Path): The path to the .pkl dataset file.
        num_examples (int | None): Cap on the number of examples. ``None`` uses the whole dataset.

    Returns:
        Dataset: The prepared Hugging Face Dataset object.
    """
    print(f"Loading dataset from {dataset_path.resolve()}...")
    df = read_pickle(dataset_path)
    dataset = Dataset.from_pandas(df)

    if num_examples is not None and num_examples < len(dataset):
        print(f"Selecting the first {num_examples} of {len(dataset)} examples...")
        dataset = dataset.select(range(num_examples))
    else:
        print(f"Using all {len(dataset)} examples.")

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
#
# The 27B/32B train in 16-bit (bf16) LoRA; the 70B/72B use 4-bit QLoRA (Unsloth 16-bit multi-GPU
# training is broken, and 4-bit ~40GB fits on one 80GB GPU). Hardware: 4x A100 80GB.
#
# `device_map`:
#   * None       -> the model fits on a single 80GB GPU (the 27B/31B models).
#   * "balanced" -> the model is too large for one GPU in 16-bit (the 70B/72B models). Unsloth
#                   splits (shards) the layers across every visible GPU (model-parallel, single
#                   process). This keeps full 16-bit precision at the cost of throughput: it is a
#                   pipeline, so only one GPU computes at a time (no data-parallel speedup).
#
# We use each model's native chat template as-is (all four ship a correct one), so there is no
# chat_template override. `response_template` gives the literal turn markers that
# train_on_responses_only masks the loss on; these match each family's native template.
#
# qwen3_6 and gemma4 are vision-language models (…ForConditionalGeneration): loaded with FastModel +
# text_only=True (skip the vision tower) and adapted via the finetune_* flags, which select the
# correct language-side attention/MLP projections. The pure-text models (qwen2_5, llama3_3) use
# FastLanguageModel with the explicit q/k/v/o + gate/up/down target modules.
#
# max_seq_length is 16384 for all: the dataset tops out at ~5k tokens (p50 ~3k), so this truncates
# nothing while staying far below the models' native context. All models use r=128 / lora_alpha=128
# (scaling 1.0). per_device_bs x accum is tuned to an effective batch of 32; a held-out eval split
# drives early stopping (~594 optimizer steps max over 2 epochs).
#
# If you have already downloaded the base models locally, set `model_id` to the local path.
MODEL_CONFIGS = {
    "qwen3": {
        "model_id": "unsloth/Qwen3-32B",
        "unsloth_class": FastLanguageModel,
        "max_seq_length": 16384,
        "device_map": None,
        "peft_settings": {
            "r": 64,
            "lora_alpha": 64,
            "lora_dropout": 0,
            "bias": "none",
            "use_gradient_checkpointing": "unsloth",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        },
        "sft_args": {
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 32,
        },
        "response_template": {
            "instruction_part": "<|im_start|>user\n",
            "response_part": "<|im_start|>assistant\n",
        },
    },
    "gemma3": {
        "model_id": "unsloth/gemma-3-27b-it",
        "unsloth_class": FastModel,
        "max_seq_length": 16384,
        "device_map": None,
        "text_only": True,  # skip the vision tower — we only train on text
        "peft_settings": {
            "finetune_vision_layers": False,
            "finetune_language_layers": True,
            "finetune_attention_modules": True,
            "finetune_mlp_modules": True,
            "r": 64,
            "lora_alpha": 64,
            "lora_dropout": 0,
            "bias": "none",
        },
        "sft_args": {
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 32,
        },
        "response_template": {
            "instruction_part": "<start_of_turn>user\n",
            "response_part": "<start_of_turn>model\n",
        },
    },
    "qwen2_5": {
        "model_id": "unsloth/Qwen2.5-72B-Instruct-bnb-4bit",
        "unsloth_class": FastLanguageModel,
        "max_seq_length": 16384,
        "device_map": None,        # 4-bit (~40GB) fits on one 80GB GPU — single-GPU QLoRA
        "load_in_4bit": True,      # Unsloth 16-bit multi-GPU training is broken; QLoRA is the supported path
        "peft_settings": {
            "r": 128,
            "lora_alpha": 128,
            "lora_dropout": 0,
            "bias": "none",
            "use_gradient_checkpointing": "unsloth",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        },
        "sft_args": {
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 32,
        },
        "response_template": {
            "instruction_part": "<|im_start|>user\n",
            "response_part": "<|im_start|>assistant\n",
        },
    },
    "llama3_3": {
        "model_id": "unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
        "unsloth_class": FastLanguageModel,
        "max_seq_length": 16384,
        "device_map": None,        # 4-bit (~40GB) fits on one 80GB GPU — single-GPU QLoRA
        "load_in_4bit": True,      # Unsloth 16-bit multi-GPU training is broken; QLoRA is the supported path
        "peft_settings": {
            "r": 128,
            "lora_alpha": 128,
            "lora_dropout": 0,
            "bias": "none",
            "use_gradient_checkpointing": "unsloth",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        },
        "sft_args": {
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 32,
        },
        "response_template": {
            "instruction_part": "<|start_header_id|>user<|end_header_id|>\n\n",
            "response_part": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        },
    },
}

def run_training(model_name: str, dataset_path: Path, script_dir: Path, resume_from_checkpoint=None):
    """
    Main function to run the fine-tuning process for a selected model.
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model '{model_name}' not recognized. Choose from: {list(MODEL_CONFIGS.keys())}")

    print(f"--- Starting fine-tuning for model: {model_name} ---")
    config = MODEL_CONFIGS[model_name]
    n_gpus = torch.cuda.device_count()
    placement = config["device_map"] or "single-GPU"
    print(f"Precision: 16-bit (bf16) | placement: {placement} | visible GPUs: {n_gpus}")

    # Define output directories relative to the script's location.
    # Per-model subdir so concurrent runs never share a checkpoint folder.
    sft_output_dir = script_dir / "baseline_outputs" / model_name
    final_lora_dir = script_dir / f"baseline_lora_adapters/{model_name}"

    # 1. Load Model and Tokenizer (27B/32B: 16-bit single-GPU; 70B/72B: 4-bit QLoRA single-GPU)
    from_pretrained_kwargs = dict(
        model_name=config["model_id"],
        max_seq_length=config["max_seq_length"],
        dtype=None,  # Auto-detection (bf16 on A100)
        load_in_4bit=config.get("load_in_4bit", False),
    )
    if config["device_map"] is not None:
        from_pretrained_kwargs["device_map"] = config["device_map"]
    if config.get("text_only"):
        from_pretrained_kwargs["text_only"] = True
    model, tokenizer = config["unsloth_class"].from_pretrained(**from_pretrained_kwargs)

    # 2. Add LoRA Adapters
    model = config["unsloth_class"].get_peft_model(model, **config["peft_settings"])

    # 3. Prepare Dataset (models use their native chat template as-is)
    dataset = load_dataset(dataset_path)

    dataset = dataset.map(
        formatting_prompts_func,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
    )

    # Hold out a small eval split to monitor for overfitting and drive early stopping.
    split = dataset.train_test_split(test_size=EVAL_SET_SIZE, seed=3407)
    train_dataset, eval_dataset = split["train"], split["test"]

    per_device_bs = config["sft_args"]["per_device_train_batch_size"]
    grad_accum = config["sft_args"]["gradient_accumulation_steps"]
    print(f"Batch: per_device={per_device_bs} x accum={grad_accum} = effective {per_device_bs * grad_accum}")
    print(f"Train/eval split: {len(train_dataset)} / {len(eval_dataset)}")

    # 4. Configure SFT Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=config["max_seq_length"],
        packing=False,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
        args=SFTConfig(
            # Core SFT arguments
            per_device_train_batch_size=per_device_bs,
            per_device_eval_batch_size=per_device_bs,
            gradient_accumulation_steps=grad_accum,
            # Shared arguments across all models
            learning_rate=1e-4,
            num_train_epochs=2,
            warmup_ratio=0.05,  # scales with total steps (~31 warmup of ~625), unlike a fixed 5
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            # Evaluate + checkpoint on the same cadence so the lowest-eval_loss adapter can be
            # restored at the end; EarlyStoppingCallback halts once eval_loss stops improving.
            eval_strategy="steps",
            eval_steps=EVAL_STEPS,
            save_strategy="steps",
            save_steps=EVAL_STEPS,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            optim="paged_adamw_32bit",
            weight_decay=0.02,
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir=str(sft_output_dir),
            report_to="none",
        ),
    )

    # 5. Set up training on responses only
    trainer = train_on_responses_only(trainer, **config["response_template"])

    # 6. Start Training (optionally resuming from a saved checkpoint)
    print(f"Starting training...{' (resuming from ' + str(resume_from_checkpoint) + ')' if resume_from_checkpoint else ''}")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
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
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint dir to resume from, or 'auto' to pick up the latest in the output dir.",
    )
    args = parser.parse_args()

    resume = args.resume_from_checkpoint
    if resume == "auto":
        resume = True

    script_dir =  Path(__file__).resolve().parent
    dataset_file_path = script_dir / "../datasets/final_datasets/baseline_sft_dataset.pkl"

    run_training(args.model_name, dataset_file_path, script_dir, resume_from_checkpoint=resume)
