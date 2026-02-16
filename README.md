# Mistral Hospitality Fine-tune

[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97%20Spaces-Demo-blue)](https://huggingface.co/spaces/Hadix10/mistral-hospitality-assistant)
[![HF Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Hadix10%2Fmistral--hospitality--qlora-orange)](https://huggingface.co/Hadix10/mistral-hospitality-qlora)

End-to-end QLoRA fine-tuning of Mistral-7B-Instruct for hospitality domain applications.

## Overview

This project demonstrates a complete ML pipeline: data preparation, QLoRA fine-tuning, evaluation, inference API, and a live Gradio demo. The model is trained on two hospitality tasks:

1. **Dialog Continuation** (SGD Hotels): Multi-turn hotel booking conversations
2. **FAQ Response** (Bitext): Single-turn hospitality intent classification and response

A third dataset (hotel reviews) was evaluated but excluded due to lack of gold labels — see the [notebook](notebooks/Mistral_QLoRA.ipynb) for the rationale.

## Results

| Metric | Value |
|--------|-------|
| **Eval Loss** | 1.38 |
| **Perplexity** | 3.97 |
| **ROUGE-1** | 0.42 |
| **ROUGE-2** | 0.21 |
| **ROUGE-L** | 0.38 |

*Evaluated on 200 held-out validation samples. Run `uv run evaluate` to reproduce.*

### Before / After Examples

**FAQ — "What are the check-in and check-out times?"**

| | Response |
|---|---|
| **Base Mistral** | Check-in and check-out times vary depending on the hotel. Generally, check-in is around 3 PM and check-out is around 11 AM. |
| **Fine-tuned** | Our standard check-in time is 3:00 PM and check-out is at 11:00 AM. Early check-in and late check-out are available upon request, subject to availability. Would you like me to arrange this for your stay? |

**Dialog — Booking continuation**

| | Response |
|---|---|
| **Base Mistral** | I can help you with that. What city are you looking to book in? |
| **Fine-tuned** | I found a great option for you — the Marriott Marquis in downtown Rome is available for May 12-14. It's a 4-star hotel with excellent reviews. Shall I go ahead and reserve a room? |

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Mistral-7B-Instruct-v0.3  (frozen, NF4 4-bit) │
│                                                 │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│   │ q_proj  │  │ k_proj  │  │ v_proj  │  Attn  │
│   │ +LoRA   │  │ +LoRA   │  │ +LoRA   │        │
│   └─────────┘  └─────────┘  └─────────┘        │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│   │gate_proj │ │ up_proj  │ │down_proj │  MLP   │
│   │ +LoRA    │ │ +LoRA    │ │ +LoRA    │        │
│   └──────────┘ └──────────┘ └──────────┘        │
│                                                 │
│   LoRA: r=8, α=16 → ~0.3% trainable params     │
│   Loss: completion-only (after [/INST])         │
└─────────────────────────────────────────────────┘
         ▲                          │
    [INST] prompt [/INST]      response</s>
         │                          ▼
┌─────────────────────────────────────────────────┐
│  Training Data                                  │
│  ├── SGD Hotels  (multi-turn dialog, ~6k)       │
│  └── Bitext      (FAQ intents, 25k)             │
│  Split per-dataset → merged → deduped           │
└─────────────────────────────────────────────────┘
```

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
git clone https://github.com/Hadix10/mistral-hospitality-finetune.git
cd mistral-hospitality-finetune
uv sync

# Train (requires GPU with ~16GB VRAM)
uv run train

# Train and push to HuggingFace Hub
uv run train --push_to_hub --hub_model_id YOUR_USERNAME/mistral-hospitality-qlora

# Evaluate
uv run evaluate --model_path Hadix10/mistral-hospitality-qlora

# Inference
uv run python src/inference.py --model_path Hadix10/mistral-hospitality-qlora --test
```

### API Server

```bash
uv sync --extra api
uv run serve
```

```bash
# Health check
curl http://localhost:8000/health

# Generate
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What are the check-in times?", "max_tokens": 256}'

# Streaming (SSE)
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Book a room for two nights", "stream": true}'
```

### Gradio Demo

```bash
python demo/app.py
# → http://localhost:7860
```

Or try the hosted version: [HF Spaces Demo](https://huggingface.co/spaces/Hadix10/mistral-hospitality-assistant)

### Colab

Open `notebooks/Mistral_QLoRA.ipynb` in Google Colab. The first cell clones this repo and installs deps from `pyproject.toml` via `uv` — no manual `pip install` needed. Set runtime to **A100** (Colab Pro).

## Project Structure

```
├── src/
│   ├── train.py          # QLoRA training pipeline
│   ├── inference.py       # Local inference + interactive mode
│   └── evaluate.py        # Perplexity + ROUGE evaluation
├── api/
│   └── main.py            # FastAPI server with SSE streaming
├── demo/
│   ├── app.py             # Gradio demo (self-contained for HF Spaces)
│   ├── requirements.txt   # HF Spaces dependencies
│   └── README.md          # HF Space metadata
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
  url = {https://github.com/Hadix10/mistral-hospitality-finetune}
}
```
