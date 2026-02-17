"""
Merge LoRA adapter weights into the base model for standalone deployment.

Why merge? The adapter is ~20MB but requires loading the full base model + PEFT
at inference time. Merging bakes the adapter into the base weights, producing a
single model that loads with plain `AutoModelForCausalLM` — no PEFT dependency.

IMPORTANT: Merging requires full-precision (bf16) base weights, NOT quantized.
The 4-bit NF4 weights used during training are lossy; merging onto them would
permanently degrade quality. We load in bf16 here, merge, then the merged model
can be quantized at serving time if needed.

Usage:
    uv run merge --adapter_path Hadix10/mistral-hospitality-qlora
    uv run merge --adapter_path ./outputs/mistral-hospitality-qlora --save_local ./merged
    uv run merge --adapter_path ./outputs/mistral-hospitality-qlora --no_push
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_and_push(
    adapter_path: str,
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.3",
    output_model_id: str = "Hadix10/mistral-hospitality-merged",
    save_local: str | None = None,
    push: bool = True,
):
    """
    Load base model in bf16, apply LoRA adapter, merge, and push to Hub.

    Args:
        adapter_path: Path to LoRA adapter (local or HF Hub).
        base_model: Base model to load adapter onto.
        output_model_id: HF Hub repo ID for the merged model.
        save_local: Optional local directory to save the merged model.
        push: Whether to push to HF Hub.
    """
    # Load tokenizer from the adapter (it may have custom tokens)
    print(f"Loading tokenizer from {adapter_path}...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model in bf16 (NOT quantized — merge requires full precision)
    print(f"Loading base model {base_model} in bf16...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    # Apply LoRA adapter
    print(f"Applying adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)

    # Merge adapter into base weights and drop LoRA layers
    print("Merging adapter weights into base model...")
    model = model.merge_and_unload()

    # Save locally if requested
    if save_local:
        print(f"Saving merged model to {save_local}...")
        model.save_pretrained(save_local)
        tokenizer.save_pretrained(save_local)
        print(f"Saved to {save_local}")

    # Push to HuggingFace Hub
    if push:
        print(f"Pushing merged model to {output_model_id}...")
        model.push_to_hub(output_model_id)
        tokenizer.push_to_hub(output_model_id)
        print(f"Pushed to https://huggingface.co/{output_model_id}")

    if not push and not save_local:
        print("Warning: --no_push set and no --save_local path. Model was merged but not saved.")

    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter into base model and push to HuggingFace Hub"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="Hadix10/mistral-hospitality-qlora",
        help="Path to LoRA adapter weights (local or HF Hub)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="Base model to merge adapter into",
    )
    parser.add_argument(
        "--output_model_id",
        type=str,
        default="Hadix10/mistral-hospitality-merged",
        help="HuggingFace Hub repo ID for the merged model",
    )
    parser.add_argument(
        "--save_local",
        type=str,
        default=None,
        help="Optional local directory to save the merged model",
    )
    parser.add_argument(
        "--no_push",
        action="store_true",
        help="Don't push to HuggingFace Hub (local save only)",
    )
    args = parser.parse_args()

    merge_and_push(
        adapter_path=args.adapter_path,
        base_model=args.base_model,
        output_model_id=args.output_model_id,
        save_local=args.save_local,
        push=not args.no_push,
    )


if __name__ == "__main__":
    main()
