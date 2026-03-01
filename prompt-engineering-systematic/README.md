# Systematic Prompt Engineering — Code from Blog Post

Extracted from: https://daddaops.com/blog/prompt-engineering-systematic/

## Scripts

| Script | Code Block | Description |
|--------|-----------|-------------|
| `eval_harness.py` | 1 | PromptEval framework — TestCase, score_classification(), PromptEval class |
| `diverse_selection.py` | 2 | Greedy max-distance diverse example selection + 3 strategy comparison |
| `prompt_versions.py` | 3 | Full V5 system prompt + prompt version comparison runner |

## Dependencies

- Python 3.8+
- All scripts are stdlib-only (dataclasses, difflib, random)

## Running

```bash
# Block 1: Test scoring logic standalone (no LLM needed)
python eval_harness.py

# Block 2: Test diversity selection algorithm (no LLM needed)
python diverse_selection.py

# Block 3: View prompt template (benchmark results require LLM API)
python prompt_versions.py
```

## Note on LLM API

The benchmark results claimed in the blog (62%→96% accuracy progression) were generated
using actual LLM API calls. Without an API key, we can verify the scoring logic and
selection algorithms work correctly, but cannot reproduce the specific accuracy numbers.
