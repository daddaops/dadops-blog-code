"""
Tool use / function calling for structured LLM output.

Demonstrates:
1. Anthropic tool use with tool_choice forcing
2. OpenAI function calling with tool_choice forcing

Requires: OPENAI_API_KEY and ANTHROPIC_API_KEY environment variables.

From: https://dadops.dev/blog/structured-output-from-llms/
"""

import json


# --- Anthropic tool use ---
def anthropic_tool_use():
    """Anthropic tool use — structured extraction via forced tool call."""
    import anthropic
    client = anthropic.Anthropic()

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        tools=[{
            "name": "extract_person",
            "description": "Extract person information from text",
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Full name"},
                    "age": {"type": "integer", "description": "Age in years"}
                },
                "required": ["name", "age"]
            }
        }],
        tool_choice={"type": "tool", "name": "extract_person"},  # force this tool
        messages=[
            {"role": "user", "content": "John Smith is 27 years old and lives in Portland."}
        ]
    )

    # The response contains a tool_use block with structured arguments
    tool_input = response.content[0].input
    print("Anthropic tool use:", tool_input)
    return tool_input


# --- OpenAI function calling ---
def openai_function_calling():
    """OpenAI function calling — structured extraction via forced function."""
    from openai import OpenAI
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "John Smith is 27 years old and lives in Portland."}
        ],
        tools=[{
            "type": "function",
            "function": {
                "name": "extract_person",
                "description": "Extract person information from text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"}
                    },
                    "required": ["name", "age"]
                }
            }
        }],
        tool_choice={"type": "function", "function": {"name": "extract_person"}}
    )

    tool_args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
    print("OpenAI function calling:", tool_args)
    return tool_args


if __name__ == "__main__":
    print("=== Tool Use / Function Calling Demos ===\n")

    print("1. Anthropic tool use:")
    try:
        anthropic_tool_use()
    except Exception as e:
        print(f"   SKIP: {e}")

    print("\n2. OpenAI function calling:")
    try:
        openai_function_calling()
    except Exception as e:
        print(f"   SKIP: {e}")
