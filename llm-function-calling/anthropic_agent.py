"""
Code Block 3: Anthropic native function calling agent loop.

From: https://dadops.dev/blog/llm-function-calling/

Complete agent loop implementation for Anthropic's native function calling.
Requires an Anthropic API key (set ANTHROPIC_API_KEY env var).

Key differences from OpenAI:
1. Stop reason is "tool_use" not "tool_calls"
2. Tool calls are in response.content blocks, mixed with text
3. Arguments in block.input are already a dict (no json.loads() needed)
4. Tool results have is_error flag and go in "user" role message
"""

import anthropic
import json
from tool_implementations import TOOLS, ANTHROPIC_TOOLS_SCHEMA


def run_agent_anthropic(user_message, tools_schema, max_turns=5):
    """Run the Anthropic agent loop with function calling."""
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": user_message}]

    for turn in range(max_turns):
        response = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=1024,
            system="You are a personal finance assistant.",
            messages=messages, tools=tools_schema,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":       # Changed: "tool_use" not "tool_calls"
            # Extract text from content blocks
            return "".join(b.text for b in response.content if b.type == "text")

        # Execute tool calls from content blocks
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            fn = TOOLS.get(block.name)
            if fn is None:
                tool_results.append({
                    "type": "tool_result", "tool_use_id": block.id,
                    "is_error": True, "content": f"Unknown function: {block.name}",
                })
                continue
            try:
                result = fn(**block.input)            # Changed: .input is already a dict
            except Exception as e:
                tool_results.append({
                    "type": "tool_result", "tool_use_id": block.id,
                    "is_error": True, "content": str(e),  # Changed: is_error flag
                })
                continue
            tool_results.append({
                "type": "tool_result", "tool_use_id": block.id,
                "content": json.dumps(result),
            })

        messages.append({"role": "user", "content": tool_results})  # Changed: role is "user"

    return "Reached max turns without a final answer."


if __name__ == "__main__":
    print("=== Anthropic Function Calling Agent ===\n")
    print("Requires ANTHROPIC_API_KEY environment variable.\n")

    query = "How much did I spend on groceries last month in euros?"
    print(f"User: {query}\n")

    try:
        answer = run_agent_anthropic(query, ANTHROPIC_TOOLS_SCHEMA)
        print(f"Assistant: {answer}")
    except anthropic.AuthenticationError:
        print("SKIP: No valid Anthropic API key found.")
    except Exception as e:
        print(f"Error: {e}")
