# Instruction Tuning from Scratch

Verified, runnable code from the [Instruction Tuning from Scratch](https://dadops.dev/blog/instruction-tuning-from-scratch/) blog post.

## Scripts

- **data_formats.py** — Alpaca and ChatML instruction data formats with statistics
- **sft_training.py** — SFT training step with instruction loss masking
- **prompt_templates.py** — ChatML, Llama-2, and Alpaca prompt template comparison
- **evaluation.py** — Before vs after SFT evaluation on 5 dimensions
- **failure_detection.py** — Diagnosing catastrophic forgetting, verbosity, and style collapse
- **landscape.py** — Landmark papers and data scaling experiment

## Run

```bash
pip install -r requirements.txt
python data_formats.py
python sft_training.py
python prompt_templates.py
python evaluation.py
python failure_detection.py
python landscape.py
```
