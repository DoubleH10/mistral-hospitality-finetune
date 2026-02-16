"""
Gradio demo for Mistral Hospitality Assistant.

Self-contained — no imports from src/ so it runs standalone on HF Spaces.

Usage:
    python demo/app.py            # local
    gradio demo/app.py            # auto-reload
"""

import os
from threading import Thread

import torch
import gradio as gr
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from peft import PeftModel

# ---------------------------------------------------------------------------
# Model loading (duplicated from src/inference.py for HF Spaces portability)
# ---------------------------------------------------------------------------
MODEL_ID = os.getenv("MODEL_ID", "Hadix10/mistral-hospitality-qlora")
BASE_MODEL = os.getenv("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")

_model = None
_tokenizer = None


def get_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    _model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    _model = PeftModel.from_pretrained(_model, MODEL_ID)
    _model.eval()
    return _model, _tokenizer


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------
MODE_PREFIXES = {
    "Auto": "",
    "Booking Dialog": "Continue this hotel booking conversation as a helpful assistant.\n\n",
    "FAQ": "",
}

MAX_CONTEXT_TOKENS = 800


def build_prompt(message: str, history: list[dict], mode: str) -> str:
    """Build [INST] prompt with sliding-window multi-turn context."""
    _, tokenizer = get_model()

    prefix = MODE_PREFIXES.get(mode, "")

    # Build conversation context from history
    turns = []
    for msg in history:
        role = msg["role"]
        text = msg["content"]
        if role == "user":
            turns.append(f"User: {text}")
        else:
            turns.append(f"Assistant: {text}")
    turns.append(f"User: {message}")

    context = "\n".join(turns)

    # Sliding window: trim from the front if too long
    tokens = tokenizer.encode(context)
    if len(tokens) > MAX_CONTEXT_TOKENS:
        context = tokenizer.decode(tokens[-MAX_CONTEXT_TOKENS:], skip_special_tokens=True)
        # Re-align to line boundary
        if "\n" in context:
            context = context[context.index("\n") + 1 :]

    instruction = f"{prefix}{context}" if prefix else context
    return f"[INST] {instruction} [/INST]"


# ---------------------------------------------------------------------------
# Streaming generation
# ---------------------------------------------------------------------------
def respond(history: list[dict], mode: str, temperature: float, max_tokens: int):
    # Last user message was appended to history by the submit handler
    message = history[-1]["content"]
    preceding = history[:-1]
    model, tokenizer = get_model()
    prompt = build_prompt(message, preceding, mode)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = {
        **inputs,
        "max_new_tokens": int(max_tokens),
        "temperature": temperature if temperature > 0 else 1.0,
        "do_sample": temperature > 0,
        "top_p": 0.9,
        "pad_token_id": tokenizer.eos_token_id,
        "streamer": streamer,
    }

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    partial = ""
    for token in streamer:
        partial += token
        yield history + [{"role": "assistant", "content": partial}]


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
CSS = """
.header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 1rem;
    color: white;
    text-align: center;
}
.header h1 { margin: 0; font-size: 1.6rem; }
.header p  { margin: 0.3rem 0 0; opacity: 0.8; font-size: 0.95rem; }

.sidebar {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 1rem;
    font-size: 0.88rem;
    line-height: 1.5;
}
.sidebar h3 { margin-top: 0; color: #1a1a2e; }
"""

# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------
EXAMPLES = [
    ["What are the check-in and check-out times?"],
    ["Is parking available at your hotel?"],
    ["I need to cancel my reservation. What's the policy?"],
    ["I'd like to book a room for two nights starting next Friday."],
    ["Can I request an extra bed in my room?"],
    ["How do I get a copy of my invoice?"],
]

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def build_ui() -> gr.Blocks:
    with gr.Blocks(css=CSS, title="Mistral Hospitality Assistant") as demo:
        # Header
        gr.HTML(
            '<div class="header">'
            "<h1>Mistral Hospitality Assistant</h1>"
            "<p>Fine-tuned Mistral-7B for hotel booking &amp; FAQ — QLoRA on SGD + Bitext</p>"
            "</div>"
        )

        with gr.Row():
            # Main chat panel
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    height=520,
                    type="messages",
                    show_copy_button=True,
                    avatar_images=(None, "https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg"),
                )
                chat_input = gr.Textbox(
                    placeholder="Ask about hotel services, bookings, policies...",
                    show_label=False,
                    container=False,
                )
                gr.Examples(
                    examples=EXAMPLES,
                    inputs=chat_input,
                )

            # Sidebar
            with gr.Column(scale=1, min_width=240):
                mode = gr.Radio(
                    choices=["Auto", "Booking Dialog", "FAQ"],
                    value="Auto",
                    label="Mode",
                )
                temperature = gr.Slider(
                    minimum=0.0, maximum=1.5, value=0.7, step=0.1,
                    label="Temperature",
                )
                max_tokens = gr.Slider(
                    minimum=32, maximum=512, value=256, step=32,
                    label="Max tokens",
                )
                gr.HTML(
                    '<div class="sidebar">'
                    "<h3>Model Info</h3>"
                    "<b>Base:</b> Mistral-7B-Instruct-v0.3<br>"
                    "<b>Method:</b> QLoRA (r=8, &alpha;=16)<br>"
                    "<b>Quant:</b> NF4 + double quant<br>"
                    "<b>Data:</b> SGD Hotels + Bitext<br>"
                    "<b>Params:</b> ~0.3% trainable<br><br>"
                    '<a href="https://huggingface.co/Hadix10/mistral-hospitality-qlora" '
                    'target="_blank">Model Card</a> · '
                    '<a href="https://github.com/Hadix10/mistral-hospitality-finetune" '
                    'target="_blank">GitHub</a>'
                    "</div>"
                )

        # Wire up chat
        chat_input.submit(
            lambda msg, hist: ("", hist + [{"role": "user", "content": msg}]),
            inputs=[chat_input, chatbot],
            outputs=[chat_input, chatbot],
        ).then(
            respond,
            inputs=[chatbot, mode, temperature, max_tokens],
            outputs=chatbot,
        )

    return demo


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)
