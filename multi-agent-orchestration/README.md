# Multi-Agent Orchestration

Verified, runnable code from the [DadOps blog post](https://dadops.co/blog/multi-agent-orchestration/).

## Scripts

| Script | Description |
|--------|-------------|
| `agent_base.py` | Base Agent class with timing and token tracking |
| `sequential_pipeline.py` | Planner → Worker → Reviewer pipeline |
| `parallel_fanout.py` | Parallel specialists with async fan-out and synthesis |
| `debate_consensus.py` | Advocate + Skeptic → Judge debate pattern |
| `budget_tracker.py` | Cost tracking and budget enforcement |

## Usage

```bash
python agent_base.py           # Single agent demo
python sequential_pipeline.py  # Sequential pipeline demo
python parallel_fanout.py      # Parallel fan-out demo
python debate_consensus.py     # Debate consensus demo
python budget_tracker.py       # Budget tracking demo
```

Note: Scripts use a mock `call_llm()` for structural verification.
Replace `llm_mock.py` with your actual API wrapper for production use.

Dependencies: none (pure Python, mock LLM calls).
