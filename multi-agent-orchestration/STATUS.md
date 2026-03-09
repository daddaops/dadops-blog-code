# multi-agent-orchestration — Homework Status

## Current Phase: EXTRACT complete

### Scripts extracted (5 scripts)
1. `agent_base.py` — Base Agent class, AgentResult, TaskState
2. `sequential_pipeline.py` — Planner → Worker → Reviewer pipeline
3. `parallel_fanout.py` — Parallel fan-out with async specialists
4. `debate_consensus.py` — Advocate + Skeptic → Judge debate
5. `budget_tracker.py` — Cost tracking with budget enforcement

### Notes
- All scripts use `call_llm()` mock from `llm_mock.py`
- Blog code is architectural patterns (API-calling), not numerical algorithms
- Mock provides deterministic responses for structural verification
