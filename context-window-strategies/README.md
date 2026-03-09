# Context Window Strategies

Code extracted from the blog post: [Context Window Strategies](https://dadops.co/blog/context-window-strategies/)

## Scripts

| Script | Description | Requires API Key |
|--------|-------------|:---:|
| `smart_truncate.py` | Priority-based truncation that scores sections by relevance and position | No |
| `chunk_and_summarize.py` | Splits documents into overlapping chunks and summarizes each with an LLM | Yes (for summarization) |
| `map_reduce.py` | Parallel map phase extracts from multiple documents, reduce phase synthesizes | Yes |
| `hierarchical_summarize.py` | Recursively summarizes until output fits a target token budget | Yes |
| `agentic_context.py` | LLM navigates a document with search/read tools, reading selectively | Yes |
| `select_strategy.py` | Decision function that picks the best strategy for given constraints | No |

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key-here"
```

## Usage

Scripts that don't require an API key run standalone demos:

```bash
python smart_truncate.py
python select_strategy.py
```

Scripts that require an API key show local demos by default and print instructions for running the full pipeline:

```bash
python chunk_and_summarize.py
python map_reduce.py
python hierarchical_summarize.py
python agentic_context.py
```
