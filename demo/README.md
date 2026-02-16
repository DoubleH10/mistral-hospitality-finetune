---
title: Mistral Hospitality Assistant
emoji: 🏨
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
hardware: t4-small
models:
  - Hadix10/mistral-hospitality-qlora
datasets:
  - vidhikatkoria/SGD_Hotels
  - bitext/Bitext-hospitality-llm-chatbot-training-dataset
---

# Mistral Hospitality Assistant

A fine-tuned Mistral-7B model for hotel booking dialogs and hospitality FAQ, running on a free T4 GPU.

**Model:** [Hadix10/mistral-hospitality-qlora](https://huggingface.co/Hadix10/mistral-hospitality-qlora)
**Code:** [GitHub](https://github.com/Hadix10/mistral-hospitality-finetune)

## Features

- Streaming responses via `TextIteratorStreamer`
- Multi-turn chat with sliding-window context (800 tokens)
- Mode selector: Auto / Booking Dialog / FAQ
- Temperature and max-tokens controls
