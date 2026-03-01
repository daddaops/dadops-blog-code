"""
Code Block 6: HybridMemoryManager — combines all memory types.

From: https://dadops.dev/blog/llm-memory-systems/

Combines sliding window + entity memory + semantic long-term memory
into one unified manager. Assembles prompts within a token budget,
giving entity memory highest priority.

Requires: numpy (via SemanticMemoryStore)
No API key required.
"""

from sliding_window import SlidingWindowMemory
from entity_memory import EntityMemory
from semantic_memory import SemanticMemoryStore


class HybridMemoryManager:
    """Combine sliding window + entity memory + semantic long-term memory."""

    def __init__(self, token_budget=3500):
        self.window = SlidingWindowMemory(recent_k=10, important_k=3)
        self.entities = EntityMemory()
        self.longterm = SemanticMemoryStore()
        self.token_budget = token_budget
        self.tpw = 1.3  # tokens per word estimate

    def _est_tokens(self, text):
        return int(len(text.split()) * self.tpw)

    def add_message(self, role, content, turn):
        """Process a new message through all memory systems."""
        self.window.add(role, content)
        if role == "user":
            self.entities.extract_and_store(content, turn)
            # In production: use LLM to decide if moment is memory-worthy
            if len(content.split()) > 15:
                self.longterm.store(content, importance=0.6)

    def assemble_prompt(self, user_message, system_prompt="You are a helpful assistant."):
        """Build the full prompt within the token budget."""
        parts = [{"role": "system", "content": system_prompt}]
        budget = self.token_budget - self._est_tokens(system_prompt)

        # 1. Entity memory (highest priority — structured facts)
        entity_ctx = self.entities.get_relevant(user_message)
        if entity_ctx:
            entity_block = f"Known facts:\n{entity_ctx}"
            cost = self._est_tokens(entity_block)
            if cost < budget * 0.15:  # cap at 15% of budget
                parts[0]["content"] += f"\n\n{entity_block}"
                budget -= cost

        # 2. Long-term memory (cross-session recall)
        lt_results = self.longterm.retrieve(user_message, top_k=3)
        if lt_results:
            lt_texts = [text for _, text, _ in lt_results]
            lt_block = "Relevant memories:\n" + "\n".join(f"- {t}" for t in lt_texts)
            cost = self._est_tokens(lt_block)
            if cost < budget * 0.15:
                parts[0]["content"] += f"\n\n{lt_block}"
                budget -= cost

        # 3. Conversation window (fills remaining budget)
        window_msgs = self.window.get_window()
        for msg in window_msgs:
            cost = self._est_tokens(msg["content"])
            if cost <= budget:
                parts.append(msg)
                budget -= cost

        # 4. Current user message
        parts.append({"role": "user", "content": user_message})
        return parts


if __name__ == "__main__":
    mgr = HybridMemoryManager(token_budget=3500)
    mgr.add_message("user", "I'm building a SaaS dashboard with React and PostgreSQL", turn=1)
    mgr.add_message("assistant", "Great stack! Let's plan the architecture.", turn=2)
    mgr.add_message("user", "I prefer pytest for testing with coverage reports", turn=3)
    mgr.add_message("assistant", "I'll set up pytest-cov in the project config.", turn=4)
    # ... many turns later ...
    mgr.add_message("user", "Now let's set up the deployment pipeline", turn=50)

    prompt = mgr.assemble_prompt("Which database should the CI pipeline test against?")
    for msg in prompt:
        role = msg["role"]
        content = msg["content"][:80] + ("..." if len(msg["content"]) > 80 else "")
        print(f"[{role}] {content}")
