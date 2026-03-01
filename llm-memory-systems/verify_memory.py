"""
Verification script for LLM Memory Systems blog post.

Tests all 6 memory implementations without requiring API keys.
"""

from datetime import datetime
from conversation_buffer import ConversationBuffer
from sliding_window import SlidingWindowMemory
from summary_memory import SummaryMemory
from entity_memory import EntityMemory
from semantic_memory import SemanticMemoryStore
from hybrid_manager import HybridMemoryManager

passed = 0
failed = 0


def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name}")


print("=" * 60)
print("LLM Memory Systems — Verification Suite")
print("=" * 60)

# --- ConversationBuffer ---
print("\n--- ConversationBuffer ---")

buf = ConversationBuffer(max_tokens=4096)
buf.add("user", "I'm building a web app with React and PostgreSQL")
buf.add("assistant", "Great stack choice! What features are you planning?")
buf.add("user", "User auth, a dashboard, and real-time notifications")

check("3 messages stored", len(buf.messages) == 3)
check("Token count > 0", buf.token_count() > 0)
check("Not near limit with 3 messages", not buf.is_near_limit())
check("get_messages returns clean format", all("tokens" not in m for m in buf.get_messages()))

# Test near-limit detection
small_buf = ConversationBuffer(max_tokens=10)
small_buf.add("user", "This is a message that should exceed the small token limit")
check("Near limit with small max_tokens", small_buf.is_near_limit())

# --- SlidingWindowMemory ---
print("\n--- SlidingWindowMemory ---")

mem = SlidingWindowMemory(recent_k=3, important_k=2)
mem.add("user", "I decide to use PostgreSQL for the database")  # high importance
mem.add("assistant", "Good choice for relational data!")
mem.add("user", "sounds good")  # low importance
mem.add("assistant", "What's next?")
mem.add("user", "Let's work on the API layer")
mem.add("assistant", "I'll set up Express routes")
mem.add("user", "Add error handling too")

window = mem.get_window()
check("Window size = 5 (3 recent + 2 important)", len(window) == 5)

# Check that the PostgreSQL decision survives
window_text = " ".join(m["content"] for m in window)
check("PostgreSQL decision survives window", "PostgreSQL" in window_text)

# Check importance scoring
pg_msg = mem.all_messages[0]
low_msg = mem.all_messages[2]
check("Decision scores higher than 'sounds good'", pg_msg["importance"] > low_msg["importance"])
check("Score capped at 1.0", all(m["importance"] <= 1.0 for m in mem.all_messages))

# Small window returns everything
small_mem = SlidingWindowMemory(recent_k=10, important_k=5)
small_mem.add("user", "hello")
check("Small set returns all messages", len(small_mem.get_window()) == 1)

# --- SummaryMemory ---
print("\n--- SummaryMemory ---")

mem = SummaryMemory(summarize_every=4)
mem.pin("User's database: PostgreSQL")
mem.add("user", "I want to build a REST API with Express")
mem.add("assistant", "Let's plan the route structure")
mem.add("user", "I need endpoints for users, posts, and comments")
mem.add("assistant", "Three resources — I'll use RESTful naming")
# After 4 messages, auto-summarization triggers
mem.add("user", "Add pagination to all list endpoints")

ctx = mem.get_context()
check("Pinned fact in context", "PostgreSQL" in ctx)
check("Summary generated after 4 turns", mem.summary != "")
check("Buffer has 1 message after summarization", len(mem.buffer) == 1)
check("Pinned section appears in context", "PINNED FACTS" in ctx)
check("Summary section appears in context", "CONVERSATION SUMMARY" in ctx)
check("Recent section appears in context", "RECENT MESSAGES" in ctx)

# Test duplicate pin prevention
mem.pin("User's database: PostgreSQL")
check("Duplicate pin prevented", len(mem.pinned) == 1)

# --- EntityMemory ---
print("\n--- EntityMemory ---")

mem = EntityMemory()
mem.extract_and_store("We're building with React and PostgreSQL", turn_number=1)
mem.extract_and_store("Authentication will use JWT tokens", turn_number=5)
mem.extract_and_store("Actually, let's switch to MySQL instead", turn_number=12)

relevant = mem.get_relevant()
check("MySQL is current database (recency wins)", "MySQL" in relevant)
check("React framework preserved", "React" in relevant)
check("JWT auth preserved", "JWT" in relevant)
check("Changelog has 1 contradiction", len(mem.get_changelog()) == 1)

changelog_entry = mem.get_changelog()[0]
check("Changelog: old value is PostgreSQL", changelog_entry["old"] == "PostgreSQL")
check("Changelog: new value is MySQL", changelog_entry["new"] == "MySQL")
check("Changelog: turn 12", changelog_entry["turn"] == 12)

# --- SemanticMemoryStore ---
print("\n--- SemanticMemoryStore ---")

store = SemanticMemoryStore()
day1 = datetime(2026, 1, 15)
day30 = datetime(2026, 2, 14)

store.store("User prefers pytest with verbose output and coverage reports",
            memory_type="preference", importance=0.8, timestamp=day1)
store.store("Project uses PostgreSQL 16 with pgvector extension",
            memory_type="decision", importance=0.9, timestamp=day1)

results = store.retrieve("How should I set up tests?", top_k=2, now=day30)
check("Retrieves 2 results", len(results) == 2)
check("Results have score, text, type", len(results[0]) == 3)

# Check that scores are positive (similarity * importance * decay)
check("Scores are numeric", all(isinstance(r[0], float) for r in results))

# Empty store returns empty
empty_store = SemanticMemoryStore()
check("Empty store returns []", empty_store.retrieve("anything") == [])

# Check embedding normalization
import numpy as np
embed = store._embed("test text")
check("Embedding is unit vector", abs(np.linalg.norm(embed) - 1.0) < 1e-6)
check("Embedding dimension correct", len(embed) == 64)

# Check deterministic embedding (seeded)
embed2 = store._embed("test text")
check("Same text → same embedding", np.allclose(embed, embed2))

# --- HybridMemoryManager ---
print("\n--- HybridMemoryManager ---")

mgr = HybridMemoryManager(token_budget=3500)
mgr.add_message("user", "I'm building a SaaS dashboard with React and PostgreSQL", turn=1)
mgr.add_message("assistant", "Great stack! Let's plan the architecture.", turn=2)
mgr.add_message("user", "I prefer pytest for testing with coverage reports", turn=3)
mgr.add_message("assistant", "I'll set up pytest-cov in the project config.", turn=4)
mgr.add_message("user", "Now let's set up the deployment pipeline", turn=50)

prompt = mgr.assemble_prompt("Which database should the CI pipeline test against?")
check("Prompt is a list", isinstance(prompt, list))
check("First message is system", prompt[0]["role"] == "system")
check("Last message is user query", prompt[-1]["role"] == "user")
check("Last message is the query", "CI pipeline" in prompt[-1]["content"])

# Entity memory should inject PostgreSQL and React
system_content = prompt[0]["content"]
check("Entity memory injected into system prompt", "PostgreSQL" in system_content or "React" in system_content)

# Check token budget is respected
total_words = sum(len(m["content"].split()) for m in prompt)
total_tokens_est = int(total_words * 1.3)
check(f"Within token budget ({total_tokens_est} est tokens < 3500)", total_tokens_est < 3500)

# --- Summary ---
print("\n" + "=" * 60)
print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
print("=" * 60)

if failed > 0:
    exit(1)
