<div align="center">
  <a href="https://ollama.com">
    <img alt="ollama" height="200px" src="https://github.com/ollama/ollama/assets/3325447/0d0b44e2-8f4a-4e99-9b52-a5c1c741c8f7">
  </a>
</div>

# Ollama (LLM4MF Fork)

This is a fork of the official [Ollama](https://github.com/ollama/ollama) repository, adapted for the LLM-Based Online Design Platform for Microfluidics.

The primary modification enhances the `/api/chat` endpoint's `format` parameter.
While standard Ollama accepts a JSON schema and converts it to a grammar internally, this version allows passing a raw GBNF (GGML-based grammar) string directly to the API. To provide a custom grammar, the `format` parameter must be a JSON string that adheres to a specific structure. This string must begin with the literal prefix `grammar:`, followed immediately by the GBNF grammar content.

This change was implemented to enforce complex, multi-part output structures, such as a free-text reasoning block (e.g., within `<think>` tags) followed by a strictly-formatted JSON object, which is not possible with a standard JSON schema alone.

---

## 🚀 Getting Started

You can either use the pre-built binary for Linux or build from the source.

### Pre-built Linux (x86_64) Version

A pre-built version for **Linux (x86_64)** is included in this repository.


1.  **Clone the Repository**: Cloning the repository will automatically download the binary files via Git LFS.
    ```shell
    git clone https://github.com/TUM-EDA/Flui3d-Chat.git
    cd "Customized Ollama"
    ```

2.  **Install and Run Ollama**: Install Ollama version 0.9.1 and replace the `ollama` binary in the installation directory (usually in `/usr/lib/ollama`) with the one from this repository. This way, you can use the standard `ollama serve` command to run the server.
---


### Run Our Fine-Tuned Microfluidic Models
To run the fine-tuned microfluidic models, ensure you have the following models pulled in your Ollama instance. You can pull them by fllowing the instructions in [Use Ollama with any GGUF Model on Hugging Face Hub](https://huggingface.co/docs/hub/en/ollama).