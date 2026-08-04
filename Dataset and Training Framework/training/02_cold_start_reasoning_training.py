import argparse
from pathlib import Path
import torch
from unsloth import FastLanguageModel, FastModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only, standardize_data_formats
from trl import SFTTrainer, SFTConfig
from datasets import Dataset, load_from_disk, concatenate_datasets
from pandas import read_pickle

# --- Training constants ---
#
# This stage CONTINUES the stage-1 adapters, which are already converged (qwen3 bottomed out at a train
# loss of ~0.01). Two consequences drive every number below:
#
#   * The learning rate must never exceed stage 1's (1e-4). Fine-tuning converged adapters at a *higher*
#     LR than the run that converged them is how you destroy them. We match 1e-4 and warm up into it —
#     stage 1 ramped over ~31 steps, so starting this run cold at full LR would hit the adapters harder
#     on step 1 than any single step of stage 1 ever did.
#   * The dataset is 90 examples, so the run is short by construction. The batch is sized to trade off
#     gradient noise (assistant turns range from ~110 words to ~900 words + JSON) against having enough
#     optimizer steps to actually install the new output format:
#
#         90 examples / effective batch 8  =~ 12 optimizer steps/epoch  x 3 epochs  =~ 36 steps
#
# There is deliberately NO eval split. Holding out 10% would cost 9 of the 60 hand-authored traces to buy
# an eval_loss over 9 examples — too noisy to early-stop on, and eval_loss is not the signal that matters
# here anyway (that signal is: does it emit a <think> block, and is the JSON still schema-clean?). Instead
# we checkpoint every epoch, so all three are on disk and can be hand-tested and compared without retraining.
SEED = 3407
REPLAY_EXAMPLES = 30   # baseline (no-<think>) examples mixed in to preserve the raw-JSON format

PER_DEVICE_BS = 1
GRAD_ACCUM = 8         # -> effective batch 8 (stage 1 used 32, but that would leave only ~3 steps/epoch here)
EPOCHS = 3
LEARNING_RATE = 1e-4   # == stage 1. NEVER raise this above stage 1's LR (see above).
WARMUP_RATIO = 0.1     # ~4 of ~36 steps; a short run needs proportionally more warmup, not less

# --- Data Loading and Formatting ---

def load_cold_start_dataset(cold_start_path: Path, final_path: Path) -> Dataset:
    """
    Loads, combines, and prepares the datasets for cold-start fine-tuning.

    The cold-start set (60 hand-authored CoT conversations) carries the *reasoning* system message and
    an assistant turn of "<think>...</think>" followed by the raw JSON. The replay slice carries the
    *baseline* system message and bare JSON with no <think> block; the two are distinguishable by their
    system turn, and the replay exists so the model does not forget the plain baseline behaviour.
    """
    print(f"Loading cold-start reasoning dataset from: {cold_start_path.resolve()}")
    try:
        cold_start_dataset = load_from_disk(str(cold_start_path))['train']
    except Exception as e:
        print(f"Error: Could not load cold-start dataset. Please check the path. Details: {e}")
        exit()

    print(f"Loading baseline dataset (replay slice) from: {final_path.resolve()}")
    try:
        df = read_pickle(final_path)
        final_dataset = Dataset.from_pandas(df)
        final_dataset = final_dataset.rename_column("conversations", "messages")
        # Select a small, random subset (seeded, so the mixture is reproducible across models/runs)
        final_dataset = final_dataset.shuffle(seed=SEED).select(range(min(REPLAY_EXAMPLES, len(final_dataset))))
    except Exception as e:
        print(f"Error: Could not load main .pkl dataset. Please check the path. Details: {e}")
        exit()

    print(f"Combining {len(cold_start_dataset)} reasoning + {len(final_dataset)} replay examples...")
    combined = concatenate_datasets([cold_start_dataset, final_dataset])
    combined = combined.shuffle(seed=SEED)

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
#
# These MUST mirror 01_baseline_training.py: this stage continues training the adapters that stage
# produced, so it has to load them under the same precision, the same max_seq_length and — above all —
# the same chat template.
#
# NO chat_template OVERRIDE. Stage 1 trains every model on its NATIVE template, so calling
# get_chat_template() here would re-render the same conversations under a *different* template than the
# adapters were built with, and the turn markers below (which train_on_responses_only masks the loss on)
# would no longer be the ones present in the text. All four native templates already round-trip an
# assistant turn of "<think>...</think>\n{json}" correctly, so the reasoning format needs nothing extra.
#
# Precision mirrors stage 1: qwen3 (32B) and gemma3 (27B) in 16-bit on one 80GB GPU; qwen2_5 (72B) and
# llama3_3 (70B) as 4-bit QLoRA (Unsloth's 16-bit multi-GPU path is broken).
#
# max_seq_length is 16384 for all, as in stage 1. The reasoning traces are long but not that long —
# measured over the actual stage-2 mixture (60 CoT + 30 replay), the *longest* rendered example is
# 6,455 tokens (gemma3; p50 ~4.0k, p95 ~5.6k), so 16384 truncates nothing and leaves ~2.5x headroom.
# Raising it would only inflate activation memory, which is already tight for the 16-bit 27B/32B runs.
# (This bounds the TRAINING examples only — inference context is set in the Ollama Modelfile.)
#
# INFERENCE NOTE (qwen3): a stage-2/3 qwen3 model must be served with THINKING ENABLED. With
# enable_thinking=False, Qwen3's template pre-fills an empty "<think>\n\n</think>" into the assistant
# turn, which collides with a model trained to emit its own reasoning block. The stage-1 baseline GGUF
# was chatted with enable_thinking=false; the reasoning models must not be.
MODEL_CONFIGS = {
    "llama3_3": {
        "model_id": "./baseline_lora_adapters/llama3_3",
        "unsloth_class": FastLanguageModel,
        "max_seq_length": 16384,
        "load_in_4bit": True,
        "response_template": {"instruction_part": "<|start_header_id|>user<|end_header_id|>\n\n", "response_part": "<|start_header_id|>assistant<|end_header_id|>\n\n"},
    },
    "qwen2_5": {
        "model_id": "./baseline_lora_adapters/qwen2_5",
        "unsloth_class": FastLanguageModel,
        "max_seq_length": 16384,
        "load_in_4bit": True,
        "response_template": {"instruction_part": "<|im_start|>user\n", "response_part": "<|im_start|>assistant\n"},
    },
    "qwen3": {
        "model_id": "./baseline_lora_adapters/qwen3",
        "unsloth_class": FastLanguageModel,
        "max_seq_length": 16384,
        "load_in_4bit": False,
        "response_template": {"instruction_part": "<|im_start|>user\n", "response_part": "<|im_start|>assistant\n"},
    },
    "gemma3": {
        "model_id": "./baseline_lora_adapters/gemma3",
        "unsloth_class": FastModel,
        "max_seq_length": 16384,
        "load_in_4bit": False,
        "text_only": True,  # VLM — skip the vision tower, as in stage 1
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
    precision = "4-bit QLoRA" if config["load_in_4bit"] else "16-bit (bf16)"
    print(f"Precision: {precision} | single-GPU | visible GPUs: {torch.cuda.device_count()}")

    # Define directories relative to the script's location.
    # Per-model checkpoint subdir so concurrent runs never share a folder (as in stage 1).
    model_path = script_dir / config["model_id"]
    sft_output_dir = script_dir / "cold_start_reasoning_outputs" / model_name
    final_lora_dir = script_dir / f"cold_start_reasoning_lora_adapters/{model_name}"

    # 1. Load Model and Tokenizer from the stage-1 adapters
    # Unsloth automatically handles loading the base model and applying the adapter
    from_pretrained_kwargs = dict(
        model_name=str(model_path),
        max_seq_length=config["max_seq_length"],
        dtype=None,
        load_in_4bit=config["load_in_4bit"],
    )
    if config.get("text_only"):
        from_pretrained_kwargs["text_only"] = True
    model, tokenizer = config["unsloth_class"].from_pretrained(**from_pretrained_kwargs)
    # NOTE: We DO NOT call get_peft_model again, as we are continuing to train the loaded adapters.

    # 2. Prepare Dataset (models use their native chat template as-is — see MODEL_CONFIGS)
    dataset = load_cold_start_dataset(cold_start_dataset_path, final_dataset_path)

    dataset = dataset.map(
        formatting_prompts_func,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
    )

    eff_batch = PER_DEVICE_BS * GRAD_ACCUM
    steps_per_epoch = -(-len(dataset) // eff_batch)  # ceil
    print(f"Batch: per_device={PER_DEVICE_BS} x accum={GRAD_ACCUM} = effective {eff_batch}")
    print(f"Schedule: {len(dataset)} examples x {EPOCHS} epochs = ~{steps_per_epoch * EPOCHS} optimizer "
          f"steps @ lr={LEARNING_RATE} (warmup {WARMUP_RATIO:.0%}, cosine)")

    # 3. Configure SFT Trainer with the new parameters
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config["max_seq_length"],
        packing=False,
        args=SFTConfig(
            per_device_train_batch_size=PER_DEVICE_BS,
            gradient_accumulation_steps=GRAD_ACCUM,
            num_train_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            warmup_ratio=WARMUP_RATIO,
            lr_scheduler_type="cosine",
            max_grad_norm=1.0,   # batches are small and the traces vary a lot in length -> spiky grads
            weight_decay=0.02,
            optim="paged_adamw_32bit",
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            # One checkpoint per epoch. The run is far shorter than the default save_steps=500, so without
            # this the run would produce NO checkpoints and picking a different epoch would mean retraining.
            save_strategy="epoch",
            save_total_limit=EPOCHS,
            seed=SEED,
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