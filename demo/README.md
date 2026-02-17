---
title: Mistral Hospitality Assistant
emoji: "\U0001F3E8"
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.23.0
app_file: app.py
pinned: false
license: mit
hardware: zero-a10g
models:
  - Hadix10/mistral-hospitality-qlora
datasets:
  - vidhikatkoria/SGD_Hotels
  - bitext/Bitext-hospitality-llm-chatbot-training-dataset
---

# Mistral Hospitality Assistant

A fine-tuned Mistral-7B model for hotel booking dialogs and hospitality FAQ, running on ZeroGPU.

**Model:** [Hadix10/mistral-hospitality-qlora](https://huggingface.co/Hadix10/mistral-hospitality-qlora)
**Code:** [GitHub](https://github.com/DoubleH10/mistral-hospitality-finetune)

## Features

- Runs your fine-tuned QLoRA model on free ZeroGPU (A10G)
- Multi-turn chat with sliding-window context (800 tokens)
- Mode selector: Auto / Booking Dialog / FAQ
- Temperature and max-tokens controls
