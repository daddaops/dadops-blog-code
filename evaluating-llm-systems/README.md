# Evaluating LLM Systems: How to Know If Your AI Actually Works

Blog post: https://daddaops.com/blog/evaluating-llm-systems/

## What's Here

A 3-level evaluation framework for LLM systems, from deterministic assertions
to LLM-as-judge to adversarial testing. All code extracted from the blog post
and organized into runnable scripts.

## Files

- `eval_structured.py` — Level 1: Structured output evaluation (field comparison, receipt parser demo)
- `eval_retrieval.py` — Level 1: Retrieval evaluation (Precision@k, Recall@k, MRR with mock retriever)
- `eval_adversarial.py` — Level 3: Adversarial test suite with mock LLM system
- `eval_harness.py` — Full EvalHarness class combining all levels, with mock components demo
- `golden_dataset.json` — Example golden dataset for testing
- `requirements.txt` — Python dependencies (none — stdlib only)

## Running

```bash
pip install -r requirements.txt
python3 eval_structured.py
python3 eval_retrieval.py
python3 eval_adversarial.py
python3 eval_harness.py
```

All scripts run with no external dependencies and no API keys.
The LLM-as-judge metric uses a mock judge function for demonstration.
