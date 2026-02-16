"""
Gradio demo for Mistral Hospitality Assistant.

Uses the HF Inference API — runs on free CPU hardware, no GPU required.
Set the HF_TOKEN secret in Space settings for gated-model access.

Usage:
    python demo/app.py            # local  (needs HF_TOKEN env var)
    gradio demo/app.py            # auto-reload
"""

import os

import gradio as gr
from huggingface_hub import InferenceClient

# ---------------------------------------------------------------------------
# Inference API client
# ---------------------------------------------------------------------------
MODEL_ID = os.getenv("MODEL_ID", "Hadix10/mistral-hospitality-qlora")
client = InferenceClient(model=MODEL_ID, token=os.getenv("HF_TOKEN"))

# ---------------------------------------------------------------------------
# Prompt formatting (matches the [INST] format used during fine-tuning)
# ---------------------------------------------------------------------------
MODE_PREFIXES = {
    "Auto": "",
    "Booking Dialog": "Continue this hotel booking conversation as a helpful assistant.\n\n",
    "FAQ": "",
}


def build_prompt(message: str, history: list[dict], mode: str) -> str:
    """Build [INST] prompt with multi-turn context."""
    prefix = MODE_PREFIXES.get(mode, "")

    turns = []
    for msg in history:
        if msg["role"] == "user":
            turns.append(f"User: {msg['content']}")
        else:
            turns.append(f"Assistant: {msg['content']}")
    turns.append(f"User: {message}")

    context = "\n".join(turns)

    # Character-level sliding window (~800 tokens ≈ 3 200 chars)
    if len(context) > 3200:
        context = context[-3200:]
        if "\n" in context:
            context = context[context.index("\n") + 1 :]

    instruction = f"{prefix}{context}" if prefix else context
    return f"[INST] {instruction} [/INST]"


# ---------------------------------------------------------------------------
# Streaming generation via Inference API
# ---------------------------------------------------------------------------
def respond(history: list[dict], mode: str, temperature: float, max_tokens: int):
    message = history[-1]["content"]
    preceding = history[:-1]
    prompt = build_prompt(message, preceding, mode)

    partial = ""
    try:
        for token in client.text_generation(
            prompt,
            max_new_tokens=int(max_tokens),
            temperature=max(temperature, 0.01),
            top_p=0.9,
            stream=True,
        ):
            partial += token
            yield history + [{"role": "assistant", "content": partial}]
    except Exception as e:
        error_msg = f"Inference error: {e}"
        yield history + [{"role": "assistant", "content": error_msg}]


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
.header h1 { margin: 0; font-size: 1.8rem; font-weight: 700; color: #ffffff; text-shadow: 0 2px 4px rgba(0,0,0,0.5); }
.header p  { margin: 0.5rem 0 0; font-size: 1rem; color: #e0e0e0; text-shadow: 0 1px 3px rgba(0,0,0,0.4); }

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
                    avatar_images=(
                        None,
                        "https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg",
                    ),
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
