"""Mock call_llm() for structural verification of orchestration patterns.

Returns deterministic responses so pipeline logic can be tested
without actual API calls.
"""
import json

_MOCK_RESPONSES = {
    "Planner": json.dumps({
        "subtasks": [
            {"id": "entity_extraction", "instruction": "Extract all named entities from the document."},
            {"id": "summarization", "instruction": "Summarize the document in 3 paragraphs."},
            {"id": "critique", "instruction": "Critique the document's methodology."},
        ]
    }),
    "Worker": "Analysis complete. Key findings identified and documented.",
    "Reviewer": "APPROVED. All subtasks completed thoroughly with consistent findings.",
    "Entity Extractor": "Entities: [Python (technology), Monte Carlo (method), Metropolis (person)]",
    "Summarizer": "The document presents Monte Carlo methods for numerical estimation. "
                  "Key techniques include importance sampling and MCMC. "
                  "Results demonstrate convergence at 1/sqrt(N) rate.",
    "Critic": "Methodology is sound but limited to synthetic benchmarks. "
              "Missing comparison with quasi-Monte Carlo methods.",
    "Synthesizer": "Merged report: 3 entities extracted, methodology validated with noted limitations.",
    "Analyst A (Advocate)": "Strong contribution: novel integration of importance sampling with MCMC. "
                            "Well-supported convergence analysis.",
    "Analyst B (Skeptic)": "Overstated novelty claim. Convergence proof assumes independence "
                           "which breaks under correlated sampling.",
    "Judge": "AGREEMENT: Core methodology is sound. "
             "DISAGREEMENT: Novelty claim needs qualification. "
             "ACTION: Add quasi-MC comparison, temper novelty language.",
}


_KEYWORD_MAP = {
    "task planner": "Planner",
    "focused analyst": "Worker",
    "quality reviewer": "Reviewer",
    "named entities": "Entity Extractor",
    "concise 3-paragraph": "Summarizer",
    "critically evaluate": "Critic",
    "multiple specialists": "Synthesizer",
    "optimistic analyst": "Analyst A (Advocate)",
    "skeptical analyst": "Analyst B (Skeptic)",
    "advocate and a skeptic": "Judge",
    "summarizer": "Summarizer",
}


def call_llm(model, system, message):
    """Mock LLM call returning (response, token_count)."""
    sys_lower = system.lower()
    for keyword, agent_name in _KEYWORD_MAP.items():
        if keyword in sys_lower:
            response = _MOCK_RESPONSES[agent_name]
            return response, len(message.split()) + len(response.split())

    # Fallback
    return "Mock response for: " + system[:50], 100
