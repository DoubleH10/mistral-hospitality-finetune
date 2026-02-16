"""
Evaluation pipeline for the fine-tuned model.

Computes:
- Perplexity (language modeling quality)
- ROUGE-L (response similarity)

Usage:
    python src/evaluate.py --model_path Hadix10/mistral-hospitality-qlora
    python src/evaluate.py --model_path ./outputs/mistral-hospitality-qlora --max_samples 100
"""

import argparse
import json
import math
import os

import torch
from tqdm import tqdm

from src.inference import generate, load_model
from src.train import Config, load_and_prepare_data


def compute_perplexity(model, tokenizer, dataset, max_samples: int = 200) -> float:
    """
    Compute perplexity on the validation set via teacher-forced cross-entropy.

    Lower is better. Measures how well the model predicts the next token.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    samples = dataset.select(range(min(max_samples, len(dataset))))

    for example in tqdm(samples, desc="Computing perplexity"):
        text = example["text"]
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])

        total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
        total_tokens += inputs["input_ids"].shape[1]

    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


def compute_rouge(
    model, tokenizer, dataset, max_samples: int = 200
) -> dict[str, float]:
    """
    Compute ROUGE scores by generating responses and comparing to references.

    Uses rouge-score library (install with `uv sync --extra eval`).
    """
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    samples = dataset.select(range(min(max_samples, len(dataset))))

    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    count = 0

    for example in tqdm(samples, desc="Computing ROUGE"):
        text = example["text"]

        # Split into prompt and reference at [/INST]
        if "[/INST]" not in text:
            continue

        prompt_part = text.split("[/INST]")[0] + "[/INST]"
        reference = text.split("[/INST]")[1].replace("</s>", "").strip()

        if not reference:
            continue

        prediction = generate(
            model, tokenizer, prompt_part,
            max_new_tokens=256, temperature=0.1, do_sample=False,
        )

        scores = scorer.score(reference, prediction)
        for key in totals:
            totals[key] += scores[key].fmeasure
        count += 1

    if count == 0:
        return {k: 0.0 for k in totals}

    return {k: round(v / count, 4) for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="Hadix10/mistral-hospitality-qlora",
        help="Path to adapter weights (local or HuggingFace Hub)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="Base model name",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=200,
        help="Max samples for each metric",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save eval_results.json",
    )
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load_model(args.model_path, args.base_model)

    # Load validation data
    print("Loading validation data...")
    config = Config()
    dataset = load_and_prepare_data(config)
    val_ds = dataset["validation"]
    print(f"Validation set: {len(val_ds)} examples")

    # Compute metrics
    print("\n--- Perplexity ---")
    ppl = compute_perplexity(model, tokenizer, val_ds, max_samples=args.max_samples)
    print(f"Perplexity: {ppl:.2f}")

    print("\n--- ROUGE ---")
    rouge = compute_rouge(model, tokenizer, val_ds, max_samples=args.max_samples)
    print(f"ROUGE-1: {rouge['rouge1']:.4f}")
    print(f"ROUGE-2: {rouge['rouge2']:.4f}")
    print(f"ROUGE-L: {rouge['rougeL']:.4f}")

    # Save results
    results = {
        "model_path": args.model_path,
        "max_samples": args.max_samples,
        "perplexity": round(ppl, 2),
        "rouge1": rouge["rouge1"],
        "rouge2": rouge["rouge2"],
        "rougeL": rouge["rougeL"],
    }

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
