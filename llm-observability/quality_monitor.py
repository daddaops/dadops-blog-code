"""
Code Block 5: Automated quality monitor with 3-signal health score.

From: https://dadops.dev/blog/llm-observability/

Three quality signals combined into a single health indicator:
1. Response length stability (recent vs baseline)
2. Semantic similarity to golden reference embeddings
3. LLM-as-judge average scores (1-5 scale)

No external dependencies required.
"""

import statistics
from collections import deque


class QualityMonitor:
    def __init__(self, golden_embeddings=None):
        self.length_history = deque(maxlen=10_000)
        self.golden_embeddings = golden_embeddings or []
        self.judge_scores = deque(maxlen=1_000)
        self.quality_log = []

    def record_response(self, response_text, embedding=None):
        word_count = len(response_text.split())
        self.length_history.append(word_count)

        sim_score = None
        if embedding and self.golden_embeddings:
            sim_score = max(
                self._cosine_sim(embedding, gold)
                for gold in self.golden_embeddings
            )
        self.quality_log.append({
            "length": word_count, "similarity": sim_score
        })

    def record_judge_score(self, score):
        """Record an LLM-as-judge quality score (1-5 scale)."""
        self.judge_scores.append(score)

    def health_score(self):
        scores = {}

        # Signal 1: length stability (recent 100 vs previous 500)
        if len(self.length_history) > 200:
            recent   = list(self.length_history)[-100:]
            baseline = list(self.length_history)[-600:-100]
            if baseline:
                drift = abs(statistics.mean(recent)
                            - statistics.mean(baseline))
                norm  = drift / (statistics.mean(baseline) + 1)
                scores["length_stability"] = max(0, 1 - norm)

        # Signal 2: semantic similarity to golden responses
        sims = [e["similarity"] for e in self.quality_log[-100:]
                if e["similarity"] is not None]
        if sims:
            scores["semantic_similarity"] = statistics.mean(sims)

        # Signal 3: LLM-as-judge average
        if self.judge_scores:
            scores["judge_score"] = (
                statistics.mean(list(self.judge_scores)[-50:]) / 5.0
            )

        # Combined health: weighted average
        if scores:
            w = {"length_stability": 0.2,
                 "semantic_similarity": 0.4,
                 "judge_score": 0.4}
            total_w  = sum(w[k] for k in scores)
            combined = sum(scores[k] * w[k] for k in scores)
            scores["overall"] = round(combined / total_w, 3)

        return scores

    @staticmethod
    def _cosine_sim(a, b):
        dot    = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x ** 2 for x in a) ** 0.5
        norm_b = sum(x ** 2 for x in b) ** 0.5
        return dot / (norm_a * norm_b + 1e-9)


if __name__ == "__main__":
    import random
    import json

    print("=== Quality Monitor ===\n")
    random.seed(42)

    # Create golden embeddings (3D for simplicity)
    golden = [[0.8, 0.5, 0.3], [0.7, 0.6, 0.4]]
    monitor = QualityMonitor(golden_embeddings=golden)

    # Simulate 300 stable responses
    for _ in range(300):
        text = " ".join(["word"] * random.randint(80, 120))
        emb = [0.75 + random.gauss(0, 0.05),
               0.55 + random.gauss(0, 0.05),
               0.35 + random.gauss(0, 0.05)]
        monitor.record_response(text, embedding=emb)
        monitor.record_judge_score(random.uniform(3.5, 5.0))

    print("After 300 stable responses:")
    scores = monitor.health_score()
    print(json.dumps(scores, indent=2))
    print()

    # Simulate quality degradation: shorter responses, drifted embeddings
    for _ in range(100):
        text = " ".join(["word"] * random.randint(40, 60))
        emb = [0.3 + random.gauss(0, 0.1),
               0.2 + random.gauss(0, 0.1),
               0.8 + random.gauss(0, 0.1)]
        monitor.record_response(text, embedding=emb)
        monitor.record_judge_score(random.uniform(2.0, 3.5))

    print("After 100 degraded responses:")
    scores = monitor.health_score()
    print(json.dumps(scores, indent=2))

    # Verify cosine similarity
    a = [1, 0, 0]
    b = [1, 0, 0]
    c = [0, 1, 0]
    print(f"\nCosine sim [1,0,0] vs [1,0,0] = {QualityMonitor._cosine_sim(a, b):.4f}")
    print(f"Cosine sim [1,0,0] vs [0,1,0] = {QualityMonitor._cosine_sim(a, c):.4f}")
