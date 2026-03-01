# Building AI Code Review Tools

Verified, runnable code from the DadOps blog post:
[Building AI Code Review Tools](https://dadops.dev/blog/building-ai-code-review-tools/)

## Scripts

| Script | Code Blocks | API Key Needed? |
|--------|------------|-----------------|
| `context_extractor.py` | 3, 5 (position mapper) | No |
| `evaluation.py` | 6 | No |
| `reviewer.py` | 1, 2, 4 | `ANTHROPIC_API_KEY` |
| `github_poster.py` | 5 | `GITHUB_TOKEN` (for posting) |

## Usage

```bash
pip install -r requirements.txt

# Runs without API keys:
python context_extractor.py     # AST extraction + diff position mapping tests
python evaluation.py            # Precision/recall/F1 evaluation framework
python github_poster.py         # Diff position mapper tests (no posting without token)

# Requires API keys:
# ANTHROPIC_API_KEY=sk-... python reviewer.py path/to/file.py
# GITHUB_TOKEN=ghp_... python github_poster.py   (posting requires token)
```

## What Each Script Does

- **context_extractor.py** — Uses Python's `ast` module to extract imports, function signatures, and class definitions from source files. Also maps line numbers to GitHub diff positions.
- **evaluation.py** — Framework for measuring code review quality (precision, recall, F1) against known-buggy code examples. Includes a mock pattern-matching reviewer for demonstration.
- **reviewer.py** — Three progressively sophisticated LLM reviewers: single-file analyzer, diff-aware PR reviewer, and multi-agent specialist pipeline (security + performance + style).
- **github_poster.py** — Posts review findings as inline comments on GitHub PRs using the Reviews API. Includes confidence filtering and diff position mapping.
