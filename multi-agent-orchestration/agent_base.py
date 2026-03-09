"""Base Agent class and AgentResult dataclass.

Demonstrates the fundamental building block: a single LLM agent
with timing, token tracking, and a defined role.
"""
from dataclasses import dataclass
import time
from typing import Any

from llm_mock import call_llm

@dataclass
class AgentResult:
    output: str
    tokens_used: int
    elapsed_sec: float
    agent_name: str

@dataclass
class Agent:
    """A single LLM agent with a defined role."""
    name: str
    system_prompt: str
    model: str = "claude-sonnet-4-20250514"

    def run(self, user_message: str) -> AgentResult:
        start = time.time()

        # call_llm() wraps your preferred provider's API
        response, tokens = call_llm(
            model=self.model,
            system=self.system_prompt,
            message=user_message,
        )

        return AgentResult(
            output=response,
            tokens_used=tokens,
            elapsed_sec=time.time() - start,
            agent_name=self.name,
        )

# Shared state that flows between agents in a pipeline
TaskState = dict[str, Any]


if __name__ == "__main__":
    agent = Agent(
        name="Summarizer",
        system_prompt="You are a summarizer. Write concise summaries.",
    )
    result = agent.run("Explain Monte Carlo methods in one paragraph.")
    print(f"Agent: {result.agent_name}")
    print(f"Output: {result.output}")
    print(f"Tokens: {result.tokens_used}")
    print(f"Time: {result.elapsed_sec:.4f}s")
