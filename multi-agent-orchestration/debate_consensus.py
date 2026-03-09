"""Debate & Consensus: Advocate + Skeptic -> Judge.

Two agents with opposing perspectives analyze the same document
independently, then a judge synthesizes their views.
"""
import asyncio

from agent_base import Agent, AgentResult

from llm_mock import call_llm  # noqa: F401


class DebateConsensus:
    """Two independent Analysts -> Judge synthesis."""

    def __init__(self):
        self.analyst_a = Agent(
            name="Analyst A (Advocate)",
            system_prompt=(
                "You are an optimistic analyst. Evaluate the document's "
                "strengths: what's novel, well-supported, and impactful. "
                "Give credit where due, but stay evidence-based."
            ),
        )
        self.analyst_b = Agent(
            name="Analyst B (Skeptic)",
            system_prompt=(
                "You are a skeptical analyst. Evaluate the document's "
                "weaknesses: what's unsupported, overstated, or missing. "
                "Be constructive but don't pull punches."
            ),
        )
        self.judge = Agent(
            name="Judge",
            system_prompt=(
                "You receive two independent analyses of the same document "
                "from an advocate and a skeptic. Your job:\n"
                "1. Identify points of AGREEMENT (high confidence)\n"
                "2. Identify points of DISAGREEMENT (needs investigation)\n"
                "3. Synthesize a balanced final assessment\n"
                "4. Produce a confidence-weighted action item list"
            ),
        )

    async def run(self, document: str) -> dict:
        prompt = f"Analyze this document:\n\n{document}"

        # Run both analysts in parallel
        loop = asyncio.get_event_loop()
        result_a, result_b = await asyncio.gather(
            loop.run_in_executor(None, self.analyst_a.run, prompt),
            loop.run_in_executor(None, self.analyst_b.run, prompt),
        )

        # Judge evaluates both
        judge_input = (
            f"## Advocate Analysis\n{result_a.output}\n\n"
            f"## Skeptic Analysis\n{result_b.output}"
        )
        judge_result = self.judge.run(
            f"Compare and synthesize these analyses:\n\n{judge_input}"
        )

        parallel_time = max(result_a.elapsed_sec, result_b.elapsed_sec)

        return {
            "result": judge_result.output,
            "agreement_analysis": judge_result.output,
            "trace": [result_a, result_b, judge_result],
            "total_tokens": (
                result_a.tokens_used
                + result_b.tokens_used
                + judge_result.tokens_used
            ),
            "total_time_sec": parallel_time + judge_result.elapsed_sec,
        }


if __name__ == "__main__":
    async def main():
        debate = DebateConsensus()
        doc = "Monte Carlo methods use random sampling to estimate numerical quantities."
        output = await debate.run(doc)

        print("=== Debate & Consensus ===")
        print(f"Result: {output['result']}")
        print(f"\nTrace ({len(output['trace'])} steps):")
        for step in output["trace"]:
            print(f"  {step.agent_name}: {step.tokens_used} tokens, {step.elapsed_sec:.4f}s")
        print(f"\nTotal tokens: {output['total_tokens']}")
        print(f"Total time (parallel-aware): {output['total_time_sec']:.4f}s")

    asyncio.run(main())
