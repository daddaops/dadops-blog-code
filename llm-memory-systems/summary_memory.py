"""
Code Block 3: SummaryMemory — compress conversation into running summaries.

From: https://dadops.dev/blog/llm-memory-systems/

Auto-summarizes every N turns, supports pinned facts that survive
summarization, and assembles context from pinned + summary + recent buffer.

No API key required (uses simulated summarization).
"""


class SummaryMemory:
    """Compress conversation into a running summary every N turns."""

    def __init__(self, summarize_every=6):
        self.buffer = []           # unsummarized messages
        self.summary = ""          # running summary document
        self.pinned = []           # facts that survive summarization
        self.summarize_every = summarize_every
        self.turn_count = 0

    def add(self, role, content):
        self.buffer.append({"role": role, "content": content})
        self.turn_count += 1

        # Auto-summarize when buffer reaches threshold
        if len(self.buffer) >= self.summarize_every:
            self._summarize()

    def pin(self, fact):
        """Pin a critical fact so it's never lost to summarization."""
        if fact not in self.pinned:
            self.pinned.append(fact)

    def _summarize(self):
        """Compress buffer into the running summary."""
        conversation = "\n".join(
            f"{m['role']}: {m['content']}" for m in self.buffer
        )
        # In production, you'd call an LLM with a prompt like:
        #   "Update the running summary with new conversation details.
        #    Current summary: {self.summary}
        #    New conversation: {conversation}
        #    Preserve all key facts, decisions, and preferences."
        self.summary = self._simulate_summary(conversation)
        self.buffer = []  # clear after summarizing

    def _simulate_summary(self, conversation):
        """Simulate LLM summarization by extracting key sentences."""
        lines = conversation.split("\n")
        key_words = {"decide", "use", "prefer", "build", "need", "want", "plan"}
        kept = []
        for line in lines:
            if any(w in line.lower() for w in key_words):
                kept.append(line.split(": ", 1)[-1] if ": " in line else line)
        new_part = ". ".join(kept[:3]) if kept else lines[0].split(": ", 1)[-1]
        return f"{self.summary} {new_part}".strip() if self.summary else new_part

    def get_context(self):
        """Assemble memory for the prompt: pinned facts + summary + recent buffer."""
        parts = []
        if self.pinned:
            parts.append("PINNED FACTS:\n" + "\n".join(f"- {f}" for f in self.pinned))
        if self.summary:
            parts.append(f"CONVERSATION SUMMARY:\n{self.summary}")
        if self.buffer:
            parts.append("RECENT MESSAGES:\n" + "\n".join(
                f"{m['role']}: {m['content']}" for m in self.buffer
            ))
        return "\n\n".join(parts)


if __name__ == "__main__":
    mem = SummaryMemory(summarize_every=4)
    mem.pin("User's database: PostgreSQL")  # pinned — never lost
    mem.add("user", "I want to build a REST API with Express")
    mem.add("assistant", "Let's plan the route structure")
    mem.add("user", "I need endpoints for users, posts, and comments")
    mem.add("assistant", "Three resources — I'll use RESTful naming")
    # ^ After 4 messages, auto-summarization triggers
    mem.add("user", "Add pagination to all list endpoints")
    print(mem.get_context())
    # Shows: pinned fact + compressed summary + recent buffer
