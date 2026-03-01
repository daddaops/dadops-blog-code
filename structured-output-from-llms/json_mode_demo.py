"""
JSON mode and JSON Schema mode for structured LLM output.

Demonstrates three approaches:
1. OpenAI basic JSON mode (response_format=json_object)
2. OpenAI JSON Schema mode (strict schema enforcement)
3. Anthropic prefill trick (start assistant response with '{')

Requires: OPENAI_API_KEY and ANTHROPIC_API_KEY environment variables.

From: https://dadops.dev/blog/structured-output-from-llms/
"""

import json

# --- Approach 1: OpenAI basic JSON mode ---
def openai_json_mode():
    """Basic JSON mode — guarantees valid JSON, no schema enforcement."""
    from openai import OpenAI
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract person info. Return JSON with: name (string), age (integer)."},
            {"role": "user", "content": "John Smith is 27 years old and lives in Portland."}
        ],
        response_format={"type": "json_object"}  # guarantees valid JSON
    )

    data = json.loads(response.choices[0].message.content)  # always parses
    print("OpenAI JSON mode:", data)
    return data


# --- Approach 2: OpenAI JSON Schema mode ---
def openai_json_schema_mode():
    """JSON Schema mode — guarantees JSON matching your exact schema."""
    from openai import OpenAI
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract person info from the text."},
            {"role": "user", "content": "John Smith is 27 years old and lives in Portland."}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "person",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"}
                    },
                    "required": ["name", "age"],
                    "additionalProperties": False
                }
            }
        }
    )

    data = json.loads(response.choices[0].message.content)
    print("OpenAI JSON Schema mode:", data)
    return data


# --- Approach 3: Anthropic prefill trick ---
def anthropic_prefill():
    """Anthropic prefill trick — start response with '{' to force JSON."""
    import anthropic
    client = anthropic.Anthropic()

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        messages=[
            {"role": "user", "content": "Extract name (string) and age (integer) as JSON.\nText: John Smith is 27."},
            {"role": "assistant", "content": "{"}  # prefill trick — force JSON start
        ]
    )

    data = json.loads("{" + response.content[0].text)  # prepend the brace we started with
    print("Anthropic prefill:", data)
    return data


if __name__ == "__main__":
    print("=== JSON Mode Demos ===\n")

    print("1. OpenAI basic JSON mode:")
    try:
        openai_json_mode()
    except Exception as e:
        print(f"   SKIP: {e}")

    print("\n2. OpenAI JSON Schema mode:")
    try:
        openai_json_schema_mode()
    except Exception as e:
        print(f"   SKIP: {e}")

    print("\n3. Anthropic prefill trick:")
    try:
        anthropic_prefill()
    except Exception as e:
        print(f"   SKIP: {e}")
