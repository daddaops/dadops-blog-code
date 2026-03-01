"""
End-to-end synthetic data generation pipeline.

Ties together all components: Self-Instruct → Few-Shot Amplification →
Evol-Instruct → Quality Filtering → Export.

Requires: OPENAI_API_KEY environment variable.

From: https://dadops.dev/blog/synthetic-data-generation/
"""

import json
from self_instruct import self_instruct
from few_shot_amplify import few_shot_amplify, SEEDS
from evol_instruct import evol_instruct
from quality_filter import quality_filter
from quality_scorecard import quality_scorecard


class SyntheticDataPipeline:
    """End-to-end: generate -> amplify -> evolve -> filter -> export."""

    def __init__(self, task_description, seed_examples=None):
        self.task = task_description
        self.seeds = seed_examples or []
        self.generated = []
        self.filtered = []

    def run(self, target_count=1000):
        # Stage 1: Bootstrap with self-instruct if no seeds
        if not self.seeds:
            print("No seeds -- bootstrapping with self-instruct...")
            self.seeds = self_instruct(self.task, target=50)

        # Stage 2: Amplify seeds to target volume
        print(f"Amplifying {len(self.seeds)} seeds...")
        amplified = few_shot_amplify(self.seeds, target=target_count)

        # Stage 3: Evolve a subset for difficulty diversity
        novel = [e for e in amplified if e.get("novelty", 0) > 0.5]
        print(f"Evolving {len(novel[:200])} examples for difficulty...")
        evolved = evol_instruct(novel[:200], rounds=2)

        self.generated = amplified + evolved
        print(f"Total raw examples: {len(self.generated)}")

        # Stage 4: Quality filter
        self.filtered = quality_filter(self.generated)
        print(f"After filtering: {len(self.filtered)}")

        # Stage 5: Quality report
        return self.filtered, quality_scorecard(self.filtered)

    def export(self, path="training_data.jsonl"):
        """Export in JSONL format for fine-tuning."""
        with open(path, "w") as f:
            for ex in self.filtered:
                f.write(json.dumps(ex) + "\n")
        print(f"Exported {len(self.filtered)} examples to {path}")


if __name__ == "__main__":
    # Run the full pipeline
    pipeline = SyntheticDataPipeline(
        task_description="Customer intent classification for e-commerce",
        seed_examples=SEEDS[:20]
    )
    dataset, scores = pipeline.run(target_count=1000)
    pipeline.export()

    # Typical output:
    # Amplifying 20 seeds...
    # Generated 1023 examples (avg novelty: 0.64)
    # Evolving 200 examples for difficulty...
    # Evolved 200 -> 487 examples
    # Total raw examples: 1510
    # After filtering: 847
    # Exported 847 examples to training_data.jsonl
