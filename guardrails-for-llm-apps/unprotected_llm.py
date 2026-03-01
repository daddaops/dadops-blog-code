"""
Unprotected LLM call — the "before" picture.

Blog post: https://dadops.dev/blog/guardrails-for-llm-apps/
Code Block 1.

REQUIRES: OpenAI API key (set OPENAI_API_KEY environment variable)

This is the vulnerable starting point — no validation, no filtering, no safety net.
"""
import openai

def ask_llm(user_message: str) -> str:
    """The 'before' picture. No validation, no filtering, no safety net."""
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful customer support agent for Acme Corp."},
            {"role": "user", "content": user_message}  # raw, unvalidated user input
        ]
    )
    return response.choices[0].message.content  # raw, unfiltered output
