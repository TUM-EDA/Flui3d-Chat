# Flui3d Chat: Dataset and Model Training

This folder contains the Python scripts and data for creating the training datasets and fine-tuning the Large Language Models (LLMs) that power the **Language-Driven Online Design Platform for Microfluidics**.

The project's primary goal is to enable the synthesis of production-ready microfluidic chip designs from diverse natural language descriptions, significantly lowering the barrier to entry for researchers and engineers. This is achieved by fine-tuning powerful base LLMs on a custom-generated synthetic dataset.

This folder covers the entire data-to-model pipeline, including:
* A multi-stage synthetic dataset generation process.
* A three-stage model adaptation strategy to train baseline and reasoning-capable models.

---

## 🔗 Related Repositories

This folder is part of a larger project. The other components can be found here:

* **Backend**: [Backend](../Backend)
* **Frontend**: [Frontend](../Frontend)
* **Modified Ollama**: [Customized Ollama](../Customized Ollama)
* **Design Synthesis Tool**: [Design Synthesis](../Backend/designsynthesis)

---

## 📋 Prerequisites

Before you begin, ensure you have the following:

* **Python**: Version 3.12, 3.11, 3.10 or 3.9.
* **PyTorch**: A recent version compatible with your hardware (CPU or GPU).
* **Dependencies**: [Install the required Python packages.](https://docs.unsloth.ai/)
* **OpenAI API Key**: Required *only* if you wish to regenerate the full synthetic dataset from scratch (Step 2 of Dataset Generation).

---

## 🗂️ Pre-Generated Dataset

**You don't need to generate the data yourself!**

This folder includes a complete, pre-generated dataset, allowing you to **skip the data generation steps** and proceed directly to **Model Training**.

The `datasets/` directory contains all final and intermediate files:

* `datasets/final_datasets/`:
    * `baseline_sft_dataset.pkl`: The final, LLM-enhanced dataset for Stage 1 (Baseline SFT).
    * `cold_start_reasoning_sft_dataset/`: The small, handcrafted dataset for Stage 2 (Cold-Start Reasoning).
    * `grpo_dataset.pkl`: The template-based dataset for Stage 3 (GRPO).
* `datasets/dataset_generation/output/openai_inputs/`: The `.jsonl` files ready to be uploaded to the OpenAI Batch API.
* `datasets/dataset_generation/output/openai_outputs/`: The corresponding `.jsonl` output files from the OpenAI Batch API.
* `datasets/dataset_generation/output/json_designs/`: The intermediate ground-truth JSON designs.

---

## 🧪 Dataset Generation (Optional)

This section is for users who want to regenerate the entire dataset from scratch. The process follows the pipeline detailed in the thesis and is divided into three steps.

### Step 1: Create Data from Scratch

This step generates the initial structured data, including random microfluidic chip topologies and template-based prompts. It produces the input files for the OpenAI API and the raw dataset used for GRPO training.

Run the following script:
```bash
python datasets/dataset_generation/01_create_data_from_scratch.py
```

1.  Generate thousands of unique microfluidic designs as graph topologies.
2.  Convert these graphs into the target JSON format.
3.  Generate prompts using three different template styles: **Connection-Oriented**, **Process-Oriented**, and **Path-Oriented (for assays)**.
4.  Save the template-based prompts and their corresponding JSON designs to `datasets/final_datasets/grpo_dataset.pkl`.
5.  Create batch input files for the OpenAI API in `datasets/dataset_generation/output/openai_inputs/`.

### Step 2: LLM-Based Prompt Augmentation

This step uses an auxiliary LLM (GPT-4o mini) to "naturalize" the template-based prompts, increasing their linguistic diversity and realism. For prompts in assay style, the auxiliary model also invents a plausible assay name as well as the names of the specific fluids or solutions—such as reagents, buffers, or sample types—used within the assay context.

1.  **Upload**: Go to the [OpenAI Batch API page](https://platform.openai.com/batches/) and upload the `.jsonl` files from the `datasets/dataset_generation/output/openai_inputs/` directory. This step requires an OpenAI account and will incur costs.
2.  **Process**: Start the batch jobs and wait for them to complete.
3.  **Download**: Download the output files for each batch job.

### Step 3: Extract Enhanced Prompts

This final step processes the output from the OpenAI Batch API and combines the newly refined prompts with their ground-truth JSON designs to create the final dataset for baseline training.

1.  **Place Files**: Move the downloaded OpenAI output files (`.jsonl`) into the corresponding subdirectories within `datasets/dataset_generation/output/openai_outputs/`.
2.  **Run Script**: Execute the extraction script:
    ```bash
    python datasets/dataset_generation/02_extract_enhanced_prompts.py
    ```
This script will parse the API responses, extract the refined prompts, pair them with the correct JSON designs, and save the final dataset as `datasets/final_datasets/baseline_sft_dataset.pkl`.

---

## 🤖 Model Training

The model adaptation process is divided into three distinct stages, moving from a general baseline model to a specialized reasoning-capable agent. We use the **Unsloth** library to accelerate fine-tuning and reduce memory usage significantly.

### Stage 1: Baseline Supervised Fine-Tuning (SFT)

This stage trains the baseline models, which can generate and modify chip designs without explicit reasoning. It uses the LLM-enhanced `baseline_sft_dataset.pkl`.

To train a model (e.g., `qwen3`), run:
```bash
python training/01_baseline_training.py qwen3
```

* **Input**: A pre-trained base model (e.g., `unsloth/Qwen3-32B`) and the `baseline_sft_dataset.pkl`.
* **Process**: Performs Parameter-Efficient Fine-Tuning (PEFT) using Low-Rank Adaptation (LoRA).
* **Output**: Saves the trained LoRA adapters to `training/baseline_lora_adapters/<model_name>/`.

### Stage 2: Cold-Start Reasoning SFT

This stage takes the baseline adapters from Stage 1 and further fine-tunes them on a small, high-quality dataset containing Chain-of-Thought (CoT) reasoning examples. This bootstraps the model's ability to "think" before providing the final JSON design.

To run this stage for the `qwen3` model, execute:
```bash
python training/02_cold_start_reasoning_training.py qwen3
```

* **Input**: The LoRA adapters from Stage 1 and the datasets in `datasets/final_datasets/cold_start_reasoning_sft_dataset/` and `baseline_sft_dataset.pkl`.
* **Process**: Continues SFT on the combined datasets.
* **Output**: Saves the new, reasoning-capable LoRA adapters to `training/cold_start_reasoning_lora_adapters/<model_name>/`.

---

### Stage 3: GRPO Reasoning Enhancement (Experimental)

This final, optional stage attempts to enhance the model's reasoning abilities using Group Relative Policy Optimization (GRPO), a reinforcement learning technique.

> **⚠️ Important Note:** This script is experimental. At the time of development, hardware limitations prevented generating a sufficiently large group of candidate responses, which is crucial for GRPO's effectiveness. Running this process is resource-intensive and may not yield optimal results without a multi-GPU setup supported by the training framework. Additionally, the provided code is untested and may crash during execution.

To run this stage for the `qwen3` model, execute:
```bash
python training/03_grpo_reasoning_training.py qwen3
```

* **Input**: The LoRA adapters from Stage 2 and the template-based `grpo_dataset.pkl`.
* **Process**: Performs GRPO training using a custom reward model to score generated designs found at `training/grpo_reward_model/`.
* **Output**: Saves the final LoRA adapters to `training/grpo_reasoning_lora_adapters/<model_name>/`.

## 🚀 Deployment with the Online Platform

To use the trained models in the **LLM4MF Online Design Platform**, the LoRA adapters must be converted to the **GGUF format**. This format is required by the project's modified Ollama instance.

The Unsloth library provides clear instructions on how to merge LoRA adapters with the base model and save them in GGUF format. **For detailed instructions, please see the official Unsloth documentation: [Saving to Ollama](https://docs.unsloth.ai/basics/running-and-saving-models/saving-to-ollama)**


**Important:** Remember to add the correct system message for the corresponding model variant to the Ollama `Modelfile`. The system messages are located in the following files:

* `datasets/resources/system_messages/baseline.txt`
* `datasets/resources/system_messages/reasoning.txt`
