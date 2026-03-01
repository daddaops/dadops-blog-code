"""
Code Block 1: ConversationBuffer — the simplest memory system.

From: https://dadops.dev/blog/llm-memory-systems/

Stores every message. O(n) growth, hits context limits fast.
The baseline that every other memory type improves upon.

No API key required.
"""


class ConversationBuffer:
    """Store every message. Simplest memory — and most expensive."""

    def __init__(self, max_tokens=4096, tokens_per_word=1.3):
        self.messages = []
        self.max_tokens = max_tokens
        self.tpw = tokens_per_word

    def add(self, role, content):
        tokens = int(len(content.split()) * self.tpw)
        self.messages.append({"role": role, "content": content, "tokens": tokens})

    def get_messages(self):
        return [{"role": m["role"], "content": m["content"]} for m in self.messages]

    def token_count(self):
        return sum(m["tokens"] for m in self.messages)

    def is_near_limit(self, threshold=0.85):
        return self.token_count() > self.max_tokens * threshold


if __name__ == "__main__":
    buf = ConversationBuffer(max_tokens=4096)
    buf.add("user", "I'm building a web app with React and PostgreSQL")
    buf.add("assistant", "Great stack choice! What features are you planning?")
    buf.add("user", "User auth, a dashboard, and real-time notifications")

    print(f"Messages: {len(buf.messages)}")        # 3
    print(f"Tokens used: {buf.token_count()}")      # ~30
    print(f"Near limit: {buf.is_near_limit()}")     # False
