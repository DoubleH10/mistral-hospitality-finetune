"""
LLM-as-Judge evaluation using Gemini.

Generates responses from the fine-tuned model and scores each one using the
Gemini API on three criteria (1-5 scale):
  - Helpfulness: Does the response address the user's need?
  - Accuracy: Are the details plausible and domain-appropriate?
  - Tone: Is the response professional and hospitality-appropriate?

This complements traditional metrics (perplexity, ROUGE) with a qualitative
assessment that better captures response quality in the hospitality domain.

Requirements:
    pip install google-genai
    export GOOGLE_API_KEY=your_key_here

Usage:
    python src/evaluate_llm.py --model_path Hadix10/mistral-hospitality-qlora
    python src/evaluate_llm.py --model_path ./outputs/mistral-hospitality-qlora --max_samples 20
"""

import argparse
import json
import os
import re
import time

from google import genai

from src.inference import generate, load_model
from src.train import Config, load_and_prepare_data

SCORING_RUBRIC = """\
You are an expert evaluator for a hospitality domain chatbot. Score the following
chatbot response on three criteria using a 1-5 scale.

## Prompt
{prompt}

## Response
{response}

## Scoring Criteria

**Helpfulness** (1-5): Does the response address the user's need? Does it provide
actionable information or move the conversation forward?
- 1: Completely irrelevant or unhelpful
- 3: Partially addresses the need but misses key aspects
- 5: Fully addresses the user's need with actionable next steps

**Accuracy** (1-5): Are the details plausible and domain-appropriate for hospitality?
Does it avoid hallucinated or contradictory information?
- 1: Contains clearly false or contradictory information
- 3: Mostly accurate but includes minor implausible details
- 5: All details are plausible and domain-appropriate

**Tone** (1-5): Is the response professional, warm, and appropriate for a hospitality
assistant? Does it match the expected register?
- 1: Rude, overly casual, or inappropriate tone
- 3: Neutral but lacks warmth or hospitality feel
- 5: Professional, warm, and perfectly suited for hospitality

Respond with ONLY a JSON object in this exact format:
{{"helpfulness": <int>, "accuracy": <int>, "tone": <int>, "reasoning": "<brief explanation>"}}
"""


def score_response(client: genai.Client, prompt: str, response: str) -> dict | None:
    """Send a (prompt, response) pair to Gemini for scoring."""
    rubric = SCORING_RUBRIC.format(prompt=prompt, response=response)

    try:
        result = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=rubric,
        )
        text = result.text.strip()

        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return None
    except Exception as e:
        print(f"  Gemini API error: {e}")
        return None


def evaluate(
    model_path: str,
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.3",
    max_samples: int = 50,
    output_path: str = "./outputs/llm_eval_results.json",
):
    """Run LLM-as-judge evaluation on validation examples."""
    # Validate API key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY environment variable is required. "
            "Get one at https://aistudio.google.com/apikey"
        )

    client = genai.Client(api_key=api_key)

    # Load fine-tuned model
    print(f"Loading model from {model_path}...")
    model, tokenizer = load_model(model_path, base_model)

    # Load validation data
    print("Loading validation data...")
    config = Config()
    dataset = load_and_prepare_data(config)
    val_ds = dataset["validation"]

    samples = val_ds.select(range(min(max_samples, len(val_ds))))
    print(f"Evaluating {len(samples)} samples...")

    # Score each response
    results = []
    for i, example in enumerate(samples):
        text = example["text"]

        if "[/INST]" not in text:
            continue

        prompt_part = text.split("[/INST]")[0] + "[/INST]"

        # Generate response from fine-tuned model
        response = generate(
            model, tokenizer, prompt_part,
            max_new_tokens=256, temperature=0.1, do_sample=False,
        )

        if not response.strip():
            continue

        # Score with Gemini
        # Extract the user-facing prompt (without [INST] wrapper) for cleaner scoring
        clean_prompt = prompt_part.replace("[INST]", "").replace("[/INST]", "").strip()
        scores = score_response(client, clean_prompt, response)

        if scores:
            scores["prompt"] = clean_prompt[:200]
            scores["response"] = response[:300]
            results.append(scores)
            print(
                f"  [{i+1}/{len(samples)}] "
                f"H={scores['helpfulness']} A={scores['accuracy']} T={scores['tone']}"
            )
        else:
            print(f"  [{i+1}/{len(samples)}] Failed to parse scores")

        # Rate limiting (Gemini free tier: 15 RPM)
        time.sleep(4)

    if not results:
        print("No results to aggregate.")
        return

    # Aggregate scores
    avg_helpfulness = sum(r["helpfulness"] for r in results) / len(results)
    avg_accuracy = sum(r["accuracy"] for r in results) / len(results)
    avg_tone = sum(r["tone"] for r in results) / len(results)
    avg_overall = (avg_helpfulness + avg_accuracy + avg_tone) / 3

    summary = {
        "model_path": model_path,
        "num_samples": len(results),
        "avg_helpfulness": round(avg_helpfulness, 2),
        "avg_accuracy": round(avg_accuracy, 2),
        "avg_tone": round(avg_tone, 2),
        "avg_overall": round(avg_overall, 2),
        "individual_results": results,
    }

    # Save results
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("LLM-as-Judge Results")
    print(f"{'='*60}")
    print(f"  Samples evaluated: {len(results)}")
    print(f"  Avg Helpfulness:   {avg_helpfulness:.2f} / 5")
    print(f"  Avg Accuracy:      {avg_accuracy:.2f} / 5")
    print(f"  Avg Tone:          {avg_tone:.2f} / 5")
    print(f"  Avg Overall:       {avg_overall:.2f} / 5")
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="LLM-as-Judge evaluation using Gemini"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="Hadix10/mistral-hospitality-qlora",
        help="Path to fine-tuned model (local or HF Hub)",
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
        default=50,
        help="Max validation samples to evaluate",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./outputs/llm_eval_results.json",
        help="Path to save evaluation results JSON",
    )
    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        base_model=args.base_model,
        max_samples=args.max_samples,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
