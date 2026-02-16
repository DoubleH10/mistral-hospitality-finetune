---
title: Mistral Hospitality Assistant
emoji: "\U0001F3E8"
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
models:
  - Hadix10/mistral-hospitality-qlora
datasets:
  - vidhikatkoria/SGD_Hotels
  - bitext/Bitext-hospitality-llm-chatbot-training-dataset
---

# Mistral Hospitality Assistant

A fine-tuned Mistral-7B model for hotel booking dialogs and hospitality FAQ, powered by the HF Inference API.

**Model:** [Hadix10/mistral-hospitality-qlora](https://huggingface.co/Hadix10/mistral-hospitality-qlora)
**Code:** [GitHub](https://github.com/Hadix10/mistral-hospitality-finetune)

## Features

- Streaming responses via HF Inference API
- Multi-turn chat with sliding-window context
- Mode selector: Auto / Booking Dialog / FAQ
- Temperature and max-tokens controls
- Runs on free CPU hardware (no GPU required)
