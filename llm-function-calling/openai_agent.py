"""
Code Block 2: OpenAI native function calling agent loop.

From: https://dadops.dev/blog/llm-function-calling/

Complete agent loop implementation for OpenAI's native function calling.
Requires an OpenAI API key (set OPENAI_API_KEY env var).

Key patterns:
- Arguments come as JSON STRING: json.loads(tc.function.arguments)
- Tool results use role: "tool" with tool_call_id
- Loop exits when msg.tool_calls is None
"""

import openai
import json
from tool_implementations import TOOLS, OPENAI_TOOLS_SCHEMA


def run_agent(user_message, tools_schema, max_turns=5):
    """Run the OpenAI agent loop with function calling."""
    client = openai.OpenAI()
    messages = [
        {"role": "system", "content": "You are a personal finance assistant."},
        {"role": "user", "content": user_message},
    ]

    for turn in range(max_turns):
        response = client.chat.completions.create(
            model="gpt-4o", messages=messages, tools=tools_schema
        )
        msg = response.choices[0].message
        messages.append(msg)  # Always append assistant message

        if msg.tool_calls is None:
            return msg.content  # No tool calls — we have the final answer

        # Execute each tool call and return results
        for tc in msg.tool_calls:
            fn = TOOLS.get(tc.function.name)
            if fn is None:
                result = {"error": f"Unknown function: {tc.function.name}"}
            else:
                try:
                    args = json.loads(tc.function.arguments)  # STRING → dict
                    result = fn(**args)
                except Exception as e:
                    result = {"error": str(e)}

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result),
            })

    return "Reached max turns without a final answer."


if __name__ == "__main__":
    print("=== OpenAI Function Calling Agent ===\n")
    print("Requires OPENAI_API_KEY environment variable.\n")

    query = "How much did I spend on groceries last month in euros?"
    print(f"User: {query}\n")

    try:
        answer = run_agent(query, OPENAI_TOOLS_SCHEMA)
        print(f"Assistant: {answer}")
    except openai.AuthenticationError:
        print("SKIP: No valid OpenAI API key found.")
    except Exception as e:
        print(f"Error: {e}")
