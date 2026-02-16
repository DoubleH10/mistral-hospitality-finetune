"""
QLoRA fine-tuning of Mistral-7B for hospitality domain.
Tasks: Dialog continuation (SGD), FAQ response (Bitext)

Usage:
    python src/train.py
    python src/train.py --push_to_hub --hub_model_id YOUR_USERNAME/mistral-hospitality-qlora
"""

import os
import argparse
import math
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


# ============================================================
# CONFIG - All hyperparameters in one place
# ============================================================
@dataclass
class Config:
    """
    Training configuration. Interview talking points:
    - r=8: Conservative rank. Increase to 16/32 if underfitting.
    - alpha=16: Scaling factor = alpha/r = 2. Standard practice.
    - target_modules: All attention + MLP projections for full adaptation.
    """
    # Model
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    
    # LoRA hyperparameters
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: tuple = field(default_factory=lambda: (
        "q_proj", "k_proj", "v_proj", "o_proj",  # attention
        "gate_proj", "up_proj", "down_proj"       # MLP
    ))
    
    # Data
    max_seq_length: int = 1024
    max_train_samples: int = 2500
    max_val_samples: int = 500
    val_split_ratio: float = 0.1
    
    # Training
    batch_size: int = 2
    gradient_accumulation_steps: int = 4  # effective batch = 8
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    warmup_ratio: float = 0.03
    
    # Output
    output_dir: str = "./outputs/mistral-hospitality-qlora"


# ============================================================
# DATA PROCESSING
# ============================================================
def clean_context(text: str, max_chars: int = 1200) -> str:
    """
    Clean and truncate conversation context, keeping most recent turns.
    
    Why truncate from the start? Recent context is more relevant for
    generating the next turn. This is a common pattern in dialog systems.
    """
    if not text:
        return ""
    text = text.replace("<SEP>", "\n")
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if len(text) > max_chars:
        text = text[-max_chars:]
        # Start at a line boundary to avoid cutting mid-sentence
        if "\n" in text:
            text = text[text.index("\n") + 1:]
    return text


def process_sgd_hotels(example: dict) -> dict:
    """
    SGD Hotels: Multi-turn dialog continuation.
    
    Dataset structure:
    - speaker: 0 = user, 1 = assistant
    - context: Previous conversation turns
    - response: Target response to generate
    
    We only train on assistant turns (speaker=1).
    """
    if example.get("speaker") != 1:
        return {"text": None}
    
    context = clean_context(example.get("context", ""))
    response = (example.get("response") or "").strip()
    
    if not context or not response:
        return {"text": None}
    
    instruction = "Continue this hotel booking conversation as a helpful assistant."
    user_msg = f"{instruction}\n\nContext:\n{context}"
    return {"text": f"[INST] {user_msg} [/INST]{response}</s>"}


def process_bitext_hospitality(example: dict) -> dict:
    """
    Bitext: Single-turn FAQ/intent response.
    
    Dataset includes intent and category metadata which we preserve
    to give the model richer training signal about query types.
    """
    instruction = (example.get("instruction") or "").strip()
    response = (example.get("response") or "").strip()
    
    if not instruction or not response:
        return {"text": None}
    
    intent = (example.get("intent") or "").strip()
    category = (example.get("category") or "").strip()
    
    if intent and category:
        instruction = f"[Category: {category} | Intent: {intent}]\n{instruction}"

    return {"text": f"[INST] {instruction} [/INST]{response}</s>"}


def load_and_prepare_data(config: Config) -> DatasetDict:
    """
    Load datasets, process, and create proper train/val splits.
    
    CRITICAL: Each dataset is split BEFORE merging to ensure:
    1. No data leakage between train and val
    2. Both tasks are represented in validation set
    """
    # --- SGD Hotels ---
    print("Loading SGD Hotels dataset...")
    sgd = load_dataset("vidhikatkoria/SGD_Hotels")["train"]
    sgd = sgd.map(process_sgd_hotels).filter(lambda x: x["text"] is not None)
    sgd = sgd.shuffle(seed=42)
    
    sgd_split = sgd.train_test_split(test_size=config.val_split_ratio, seed=42)
    sgd_train, sgd_val = sgd_split["train"], sgd_split["test"]
    print(f"  SGD: {len(sgd_train)} train, {len(sgd_val)} val")
    
    # --- Bitext Hospitality ---
    print("Loading Bitext Hospitality dataset...")
    bitext = load_dataset("bitext/Bitext-hospitality-llm-chatbot-training-dataset")["train"]
    bitext = bitext.map(process_bitext_hospitality).filter(lambda x: x["text"] is not None)
    bitext = bitext.shuffle(seed=42)
    
    bitext_split = bitext.train_test_split(test_size=config.val_split_ratio, seed=42)
    bitext_train, bitext_val = bitext_split["train"], bitext_split["test"]
    print(f"  Bitext: {len(bitext_train)} train, {len(bitext_val)} val")
    
    # --- Merge and cap ---
    train_ds = concatenate_datasets([sgd_train, bitext_train]).shuffle(seed=42)
    val_ds = concatenate_datasets([sgd_val, bitext_val]).shuffle(seed=42)
    
    if len(train_ds) > config.max_train_samples:
        train_ds = train_ds.select(range(config.max_train_samples))
    if len(val_ds) > config.max_val_samples:
        val_ds = val_ds.select(range(config.max_val_samples))
    
    print(f"Final dataset: {len(train_ds)} train, {len(val_ds)} val")
    
    return DatasetDict({"train": train_ds, "validation": val_ds})


# ============================================================
# MODEL SETUP
# ============================================================
def load_model_and_tokenizer(config: Config):
    """
    Load base model with 4-bit quantization and attach LoRA adapters.
    
    Key concepts for interviews:
    - NF4: NormalFloat4, optimized for normally-distributed weights
    - Double quantization: Quantize the quantization constants too
    - Compute dtype bfloat16: Dequantize to bf16 during forward pass
    """
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Required for causal LM training
    
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    # Load model
    print(f"Loading {config.base_model} with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False  # Incompatible with gradient checkpointing
    
    # Prepare for QLoRA training
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    # Attach LoRA adapters
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(config.target_modules),
    )
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model, tokenizer


# ============================================================
# TRAINING
# ============================================================
def train(
    config: Config,
    push_to_hub: bool = False,
    hub_model_id: Optional[str] = None
):
    """
    Main training function.
    
    Uses DataCollatorForCompletionOnlyLM to compute loss only on
    response tokens, not on instruction/context. This is standard
    practice for instruction-tuning.
    """
    # Load data
    dataset = load_and_prepare_data(config)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Collator: Only compute loss on response tokens (after [/INST])
    collator = DataCollatorForCompletionOnlyLM(
        response_template="[/INST]",
        tokenizer=tokenizer,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        bf16=True,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        max_seq_length=config.max_seq_length,
        dataset_text_field="text",
    )
    
    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    trainer.train()
    
    # Save final model
    print(f"\nSaving model to {config.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)
    
    # Print final metrics
    metrics = trainer.evaluate()
    ppl = math.exp(metrics["eval_loss"])
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"  Final eval loss: {metrics['eval_loss']:.4f}")
    print(f"  Perplexity: {ppl:.2f}")
    print(f"  Model saved to: {config.output_dir}")
    print("="*60 + "\n")
    
    return trainer


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Mistral-7B on hospitality data using QLoRA"
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push trained adapter to HuggingFace Hub"
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="HuggingFace Hub model ID (e.g., username/model-name)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override default output directory"
    )
    args = parser.parse_args()
    
    # Initialize config
    config = Config()
    
    if args.output_dir:
        config.output_dir = args.output_dir
    
    if args.push_to_hub and not args.hub_model_id:
        raise ValueError("--hub_model_id required when --push_to_hub is set")
    
    # Run training
    train(config, push_to_hub=args.push_to_hub, hub_model_id=args.hub_model_id)


if __name__ == "__main__":
    main()
