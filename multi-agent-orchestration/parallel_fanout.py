"""Parallel Fan-Out: Router -> parallel Specialists -> Synthesizer.

Demonstrates concurrent agent execution where independent specialists
run in parallel and a synthesizer merges their results.
"""
import asyncio

from agent_base import Agent, AgentResult

from llm_mock import call_llm  # noqa: F401


class ParallelFanOut:
    """Router -> parallel Specialists -> Synthesizer."""

    def __init__(self):
        self.specialists = {
            "entities": Agent(
                name="Entity Extractor",
                system_prompt=(
                    "Extract all named entities (people, organizations, "
                    "technologies, metrics) from the document. Return a "
                    "structured list with entity type and context."
                ),
            ),
            "summary": Agent(
                name="Summarizer",
                system_prompt=(
                    "Write a concise 3-paragraph summary of the document's "
                    "key findings, methodology, and conclusions."
                ),
            ),
            "critique": Agent(
                name="Critic",
                system_prompt=(
                    "Critically evaluate the document's methodology, "
                    "identify weaknesses, unsupported claims, and "
                    "suggest improvements. Be specific and constructive."
                ),
            ),
        }
        self.synthesizer = Agent(
            name="Synthesizer",
            system_prompt=(
                "You receive analyses from multiple specialists. Merge "
                "them into a single coherent report. Resolve any conflicts "
                "by noting the disagreement. Structure: Entities, Summary, "
                "Critique, Action Items."
            ),
        )

    async def _run_specialist(self, name, agent, document):
        """Run a single specialist, catching failures gracefully."""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, agent.run, f"Analyze this document:\n\n{document}"
            )
            return name, result, None
        except Exception as e:
            return name, None, str(e)

    async def run(self, document: str) -> dict:
        # Fan out to all specialists concurrently
        tasks = [
            self._run_specialist(name, agent, document)
            for name, agent in self.specialists.items()
        ]
        results = await asyncio.gather(*tasks)

        # Collect outputs, handling partial failures
        trace = []
        specialist_outputs = {}
        failures = []
        for name, result, error in results:
            if error:
                failures.append(f"{name}: {error}")
            else:
                trace.append(result)
                specialist_outputs[name] = result.output

        if failures:
            specialist_outputs["_failures"] = failures

        # Synthesize
        combined = "\n\n".join(
            f"## {name}\n{out}"
            for name, out in specialist_outputs.items()
            if not name.startswith("_")
        )
        if failures:
            combined += f"\n\nNote: these specialists failed: {failures}"

        synth_result = self.synthesizer.run(
            f"Merge these specialist analyses:\n\n{combined}"
        )
        trace.append(synth_result)

        return {
            "result": synth_result.output,
            "trace": trace,
            "failures": failures,
            "total_tokens": sum(r.tokens_used for r in trace),
            "total_time_sec": max(
                r.elapsed_sec for r in trace[:-1]  # parallel portion
            ) + trace[-1].elapsed_sec,  # plus synthesizer
        }


if __name__ == "__main__":
    async def main():
        fanout = ParallelFanOut()
        doc = "Monte Carlo methods use random sampling to estimate numerical quantities."
        output = await fanout.run(doc)

        print("=== Parallel Fan-Out ===")
        print(f"Result: {output['result']}")
        print(f"Failures: {output['failures']}")
        print(f"\nTrace ({len(output['trace'])} steps):")
        for step in output["trace"]:
            print(f"  {step.agent_name}: {step.tokens_used} tokens, {step.elapsed_sec:.4f}s")
        print(f"\nTotal tokens: {output['total_tokens']}")
        print(f"Total time (parallel-aware): {output['total_time_sec']:.4f}s")

    asyncio.run(main())
