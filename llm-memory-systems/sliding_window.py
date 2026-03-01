"""
Code Block 2: SlidingWindowMemory — keep recent + important older messages.

From: https://dadops.dev/blog/llm-memory-systems/

Uses a heuristic importance scorer to decide which old messages survive
the sliding window. Questions, decisions, and code score highest.

No API key required.
"""


class SlidingWindowMemory:
    """Keep recent messages + high-importance older ones."""

    def __init__(self, recent_k=10, important_k=5):
        self.all_messages = []
        self.recent_k = recent_k
        self.important_k = important_k

    def _score(self, msg):
        """Heuristic importance: questions > decisions > code > chat."""
        text = msg["content"].lower()
        score = 0.1  # baseline
        if "?" in msg["content"]:
            score += 0.3
        if any(kw in text for kw in ["decide", "use", "prefer", "choose", "switch"]):
            score += 0.4
        if any(kw in text for kw in ["def ", "class ", "import ", "select ", "create "]):
            score += 0.25
        if len(text.split()) > 30:
            score += 0.15  # longer messages tend to carry more info
        return min(score, 1.0)

    def add(self, role, content):
        msg = {"role": role, "content": content, "turn": len(self.all_messages)}
        msg["importance"] = self._score(msg)
        self.all_messages.append(msg)

    def get_window(self):
        if len(self.all_messages) <= self.recent_k + self.important_k:
            return self.all_messages  # everything fits

        recent = self.all_messages[-self.recent_k:]
        older = self.all_messages[:-self.recent_k]

        # Top important_k from older messages, by importance score
        important = sorted(older, key=lambda m: -m["importance"])[:self.important_k]

        # Merge and sort by original turn order
        merged = sorted(important + recent, key=lambda m: m["turn"])
        return [{"role": m["role"], "content": m["content"]} for m in merged]


if __name__ == "__main__":
    mem = SlidingWindowMemory(recent_k=3, important_k=2)
    mem.add("user", "I decide to use PostgreSQL for the database")  # high importance
    mem.add("assistant", "Good choice for relational data!")
    mem.add("user", "sounds good")                                   # low importance
    mem.add("assistant", "What's next?")
    mem.add("user", "Let's work on the API layer")
    mem.add("assistant", "I'll set up Express routes")
    mem.add("user", "Add error handling too")

    window = mem.get_window()
    print(f"Window size: {len(window)}")  # 5 (3 recent + 2 important)
    for msg in window:
        print(f"  {msg['role']}: {msg['content'][:60]}")
