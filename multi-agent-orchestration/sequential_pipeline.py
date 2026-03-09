"""Sequential Pipeline: Planner -> Workers -> Reviewer.

Demonstrates the factory assembly line pattern where a planner
decomposes work, workers execute subtasks, and a reviewer validates.
"""
import json

from agent_base import Agent, AgentResult

from llm_mock import call_llm  # noqa: F401 (used by Agent internally)


class SequentialPipeline:
    """Planner -> Workers -> Reviewer pipeline."""

    def __init__(self):
        self.planner = Agent(
            name="Planner",
            system_prompt=(
                "You are a task planner. Given a document analysis request, "
                "break it into exactly 3 subtasks: entity_extraction, "
                "summarization, and critique. Return JSON: "
                '{"subtasks": [{"id": "...", "instruction": "..."}]}'
            ),
        )
        self.worker = Agent(
            name="Worker",
            system_prompt=(
                "You are a focused analyst. Complete ONLY the specific "
                "subtask you are given. Be thorough but stay on-task. "
                "Do not address other aspects of the document."
            ),
        )
        self.reviewer = Agent(
            name="Reviewer",
            system_prompt=(
                "You are a quality reviewer. Check the combined analysis "
                "for completeness, accuracy, and consistency. List any "
                "issues found. If satisfactory, respond with APPROVED "
                "followed by a final merged summary."
            ),
        )

    def run(self, document: str) -> dict:
        trace = []

        # Step 1: Plan
        plan_result = self.planner.run(
            f"Create subtasks for analyzing this document:\n\n{document}"
        )
        trace.append(plan_result)
        subtasks = json.loads(plan_result.output)["subtasks"]

        # Step 2: Execute each subtask with a focused worker
        worker_outputs = {}
        for task in subtasks:
            result = self.worker.run(
                f"Document:\n{document}\n\n"
                f"Your task: {task['instruction']}"
            )
            trace.append(result)
            worker_outputs[task["id"]] = result.output

        # Step 3: Review the combined output
        combined = "\n\n".join(
            f"## {tid}\n{out}" for tid, out in worker_outputs.items()
        )
        review = self.reviewer.run(
            f"Review this combined analysis:\n\n{combined}"
        )
        trace.append(review)

        total_tokens = sum(r.tokens_used for r in trace)
        total_time = sum(r.elapsed_sec for r in trace)

        return {
            "result": review.output,
            "trace": trace,
            "total_tokens": total_tokens,
            "total_time_sec": total_time,
        }


if __name__ == "__main__":
    pipeline = SequentialPipeline()
    doc = "Monte Carlo methods use random sampling to estimate numerical quantities."
    output = pipeline.run(doc)

    print("=== Sequential Pipeline ===")
    print(f"Result: {output['result']}")
    print(f"\nTrace ({len(output['trace'])} steps):")
    for step in output["trace"]:
        print(f"  {step.agent_name}: {step.tokens_used} tokens, {step.elapsed_sec:.4f}s")
    print(f"\nTotal tokens: {output['total_tokens']}")
    print(f"Total time: {output['total_time_sec']:.4f}s")
