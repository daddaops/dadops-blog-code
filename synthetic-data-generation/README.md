# Synthetic Data Generation — Verified Code

Runnable code from the DadOps blog post: [Synthetic Data Generation: Using LLMs to Build Your Own Training Datasets](https://daddaops.com/blog/synthetic-data-generation/)

## Scripts

| Script | Description | API Key Required? |
|--------|-------------|-------------------|
| `self_instruct.py` | Generate training data from task description alone | Yes (OPENAI_API_KEY) |
| `few_shot_amplify.py` | Scale seed examples via few-shot prompting | Yes (OPENAI_API_KEY) |
| `evol_instruct.py` | Evolve simple examples into harder variants | Yes (OPENAI_API_KEY) |
| `quality_filter.py` | 4-stage quality filter pipeline | Yes (OPENAI_API_KEY) |
| `quality_scorecard.py` | Compute quality metrics for a synthetic dataset | Yes (OPENAI_API_KEY) |
| `pipeline.py` | End-to-end pipeline tying all components together | Yes (OPENAI_API_KEY) |

## Quick Start

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your-key-here

# Run individual components
python self_instruct.py
python few_shot_amplify.py
python evol_instruct.py

# Run the full pipeline
python pipeline.py
```

## Notes

- All scripts require an OpenAI API key for LLM generation
- `quality_filter.py` and `quality_scorecard.py` also use the OpenAI embeddings API
- The comparison table in the blog uses estimated ranges (not exact benchmarks)
- Cost estimates are approximate based on Feb 2026 pricing
