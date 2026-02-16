# Mistral Hospitality Fine-tune

End-to-end QLoRA fine-tuning of Mistral-7B-Instruct for hospitality domain applications.

## Overview

This project demonstrates a complete ML pipeline: data preparation, QLoRA fine-tuning, evaluation, and inference. The model is trained on two hospitality tasks:

1. **Dialog Continuation** (SGD Hotels): Multi-turn hotel booking conversations
2. **FAQ Response** (Bitext): Single-turn hospitality intent classification and response

A third dataset (hotel reviews) was evaluated but excluded due to lack of gold labels — see the [notebook](notebooks/Mistral_QLoRA.ipynb) for the rationale.

## Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| **QLoRA (r=8, α=16)** | Memory-efficient fine-tuning. 4-bit base weights + bf16 adapters. ~0.3% trainable params |
| **NF4 quantization** | NormalFloat4 optimized for normally-distributed neural network weights |
| **Completion-only loss** | `DataCollatorForCompletionOnlyLM` — loss computed only on response tokens after `[/INST]` |
| **Per-dataset splits** | Each dataset split independently with `train_test_split()` before merging — zero train/val overlap |
| **Mistral `[INST]` template** | Native chat format for fair base-vs-adapter comparison |

## Quick Start

```bash
# Clone and setup
git clone https://github.com/DoubleH10/mistral-hospitality-finetune.git
cd mistral-hospitality-finetune
uv sync

# Train (requires GPU with ~16GB VRAM)
uv run python src/train.py

# Train and push to HuggingFace Hub
uv run python src/train.py --push_to_hub --hub_model_id YOUR_USERNAME/mistral-hospitality-qlora

# Inference
uv run python src/inference.py --model_path ./outputs/mistral-hospitality-qlora --test
```

### Colab

Open `notebooks/Mistral_QLoRA.ipynb` in Google Colab. The first cell clones this repo and installs deps from `pyproject.toml` via `uv` — no manual `pip install` needed. Set runtime to **A100** (Colab Pro).

## Project Structure

```
├── src/
│   ├── train.py          # QLoRA training pipeline (production)
│   ├── inference.py       # Local inference + interactive mode
│   └── evaluate.py        # Evaluation pipeline (TODO)
├── api/
│   └── main.py            # FastAPI inference endpoint (TODO)
├── notebooks/
│   └── Mistral_QLoRA.ipynb  # Exploration notebook (Colab-ready)
├── outputs/               # Training artifacts (.gitignored)
├── pyproject.toml         # Single source of truth for deps
└── uv.lock                # Pinned dependency versions
```

## Training Details

**Base Model:** `mistralai/Mistral-7B-Instruct-v0.3`

**Datasets:**
- [SGD Hotels](https://huggingface.co/datasets/vidhikatkoria/SGD_Hotels) — Schema-guided multi-turn dialog (~6k examples)
- [Bitext Hospitality](https://huggingface.co/datasets/bitext/Bitext-hospitality-llm-chatbot-training-dataset) — Intent-response pairs (25k examples)

**Hyperparameters:**
```
LoRA rank (r)       8         Low-rank dimension
LoRA alpha          16        Scaling factor (α/r = 2)
Learning rate       2e-4      Standard for QLoRA
Batch size          2         Per device
Gradient accum      4         Effective batch = 8
Epochs              3
Optimizer           paged_adamw_8bit
Scheduler           cosine with 3% warmup
```

**Hardware:** Tested on Google Colab A100 (40GB).

## Model Card

**Intended Use:** Hospitality chatbots, hotel booking assistants, customer service automation

**Limitations:**
- Domain-specific (hospitality only)
- English language only
- Not suitable for safety-critical applications without human oversight

**Training Data:** Publicly available datasets. No PII.

## License

MIT

## Citation

```bibtex
@misc{hijazi2026mistral-hospitality,
  author = {Hijazi, Hadi},
  title = {Mistral Hospitality Fine-tune: QLoRA for Hotel Domain},
  year = {2026},
  url = {https://github.com/DoubleH10/mistral-hospitality-finetune}
}
```
