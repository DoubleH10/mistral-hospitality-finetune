"""
Local inference with the fine-tuned model.

Usage:
    python src/inference.py --model_path ./outputs/mistral-hospitality-qlora
    python src/inference.py --model_path username/mistral-hospitality-qlora  # From Hub
"""

import argparse
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def load_model(
    model_path: str,
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.3",
    load_in_4bit: bool = True,
):
    """
    Load the fine-tuned model with adapter weights.
    
    Args:
        model_path: Path to adapter weights (local or HF Hub)
        base_model: Base model to load adapters onto
        load_in_4bit: Whether to quantize base model
    
    Returns:
        model, tokenizer tuple
    """
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Quantization config (same as training)
    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    
    print(f"Loading base model {base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print(f"Loading adapter from {model_path}...")
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()
    
    return model, tokenizer


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    do_sample: bool = True,
    top_p: float = 0.9,
) -> str:
    """
    Generate a response from the model.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        prompt: Input prompt (will be formatted if needed)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (higher = more random)
        do_sample: Whether to sample or use greedy decoding
        top_p: Nucleus sampling parameter
    
    Returns:
        Generated text (response only, not including prompt)
    """
    # Format prompt if not already in Mistral's [INST] template
    if "[INST]" not in prompt:
        prompt = f"[INST] {prompt} [/INST]"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            do_sample=do_sample,
            top_p=top_p if do_sample else 1.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode and extract response (text after [/INST])
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "[/INST]" in full_output:
        response = full_output.split("[/INST]")[-1].strip()
    else:
        response = full_output[len(prompt):].strip()

    return response


def format_dialog_prompt(context: str) -> str:
    """Format a multi-turn dialog prompt using Mistral's [INST] template."""
    instruction = "Continue this hotel booking conversation as a helpful assistant."
    return f"[INST] {instruction}\n\nContext:\n{context} [/INST]"


def format_faq_prompt(question: str, category: Optional[str] = None, intent: Optional[str] = None) -> str:
    """Format a single-turn FAQ prompt using Mistral's [INST] template."""
    if category and intent:
        question = f"[Category: {category} | Intent: {intent}]\n{question}"
    return f"[INST] {question} [/INST]"


def interactive_mode(model, tokenizer):
    """Run interactive chat loop."""
    print("\n" + "="*60)
    print("Interactive Mode - Type 'quit' to exit")
    print("="*60 + "\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            if not user_input:
                continue
            
            response = generate(model, tokenizer, user_input)
            print(f"Assistant: {response}\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def run_test_prompts(model, tokenizer):
    """Run standard test prompts to evaluate model."""
    test_prompts = [
        # FAQ style
        "What are your check-in and check-out times?",
        "Do you have parking available?",
        "Can I cancel my reservation?",
        
        # Dialog style (with context)
        format_dialog_prompt(
            "User: I need a room in Rome for 2 nights next week.\n"
            "Assistant: Certainly! What dates are you looking at?\n"
            "User: Arriving May 12th, leaving May 14th."
        ),
        
        # Intent-specific
        format_faq_prompt(
            "I want to change my booking to a different date",
            category="booking",
            intent="modify_booking"
        ),
    ]
    
    print("\n" + "="*60)
    print("Running Test Prompts")
    print("="*60)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i} ---")
        print(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")
        response = generate(model, tokenizer, prompt, do_sample=False)
        print(f"Response: {response}")


def main():
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./outputs/mistral-hospitality-qlora",
        help="Path to adapter weights (local or HuggingFace Hub)"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="Base model name"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive chat mode"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test prompts"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to run"
    )
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.model_path, args.base_model)
    
    if args.interactive:
        interactive_mode(model, tokenizer)
    elif args.test:
        run_test_prompts(model, tokenizer)
    elif args.prompt:
        response = generate(model, tokenizer, args.prompt)
        print(f"Response: {response}")
    else:
        # Default: run test prompts then interactive
        run_test_prompts(model, tokenizer)
        interactive_mode(model, tokenizer)


if __name__ == "__main__":
    main()
