# Fine-Tuning LLMs: When Prompt Engineering Isn't Enough

Blog post: https://daddaops.com/blog/fine-tuning-llms/

## What's Here

A complete fine-tuning pipeline for customer support ticket classification,
from data preparation to evaluation. Includes both OpenAI API and open-source
LoRA approaches.

## Files

- `training_data.jsonl` — Example training data (3 examples from blog)
- `validate_training_data.py` — Pre-flight validation of JSONL training data (no API key needed)
- `bootstrap_dataset.py` — Generate training data at scale using GPT-4o (requires API key)
- `fine_tune_openai.py` — Fine-tune via OpenAI API (requires API key)
- `fine_tune_lora.py` — Fine-tune open-source Llama with QLoRA (requires GPU)
- `evaluate_models.py` — Compare base vs fine-tuned model (requires API key)
- `verify_roi_calculator.py` — Verify the blog's ROI calculator math (no API key needed)

## Running

Scripts that don't need API keys:
```bash
python3 validate_training_data.py
python3 verify_roi_calculator.py
```

Scripts that need OpenAI API key:
```bash
export OPENAI_API_KEY=your-key
python3 bootstrap_dataset.py
python3 fine_tune_openai.py
python3 evaluate_models.py
```

LoRA fine-tuning (needs GPU + packages):
```bash
pip install transformers peft trl bitsandbytes
python3 fine_tune_lora.py
```
