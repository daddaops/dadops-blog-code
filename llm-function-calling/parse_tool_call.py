"""
Code Block 1: Regex-based tool call parsing (the fragile pre-API approach).

From: https://dadops.dev/blog/llm-function-calling/

This demonstrates the old approach to function calling before native API support.
It works ~80% of the time with GPT-4 class models but fails on malformed XML/JSON,
hallucinated function names, and tool calls embedded in conversational text.

No API key required — this is just a parser.
"""

import re
import json


SYSTEM_PROMPT = """You have access to these tools:

<tools>
  <tool name="query_transactions">
    <description>Search transaction history</description>
    <params>
      <param name="category" type="string"/>
      <param name="month" type="string">YYYY-MM format</param>
    </params>
  </tool>
</tools>

When you need a tool, respond ONLY with:
<tool_call name="...">{"key": "value"}</tool_call>
"""


def parse_tool_call(text):
    """Extract tool call from raw model output. Fragile!"""
    match = re.search(
        r'<tool_call name="(\w+)">(.*?)</tool_call>',
        text, re.DOTALL
    )
    if not match:
        return None  # Model didn't follow the format
    try:
        return match.group(1), json.loads(match.group(2).strip())
    except json.JSONDecodeError:
        return None  # Invalid JSON — trailing commas, single quotes...


# What goes wrong:
# 1. Model invents tools: <tool_call name="send_money">...
# 2. Malformed XML: missing closing tags, extra whitespace
# 3. Broken JSON: {category: 'food'} instead of {"category": "food"}
# 4. Unnecessary calls: "Sure! Let me check. <tool_call..."


if __name__ == "__main__":
    # Test cases
    print("=== Regex Tool Call Parser Tests ===\n")

    # Good input
    good = '<tool_call name="query_transactions">{"category": "groceries", "month": "2026-01"}</tool_call>'
    result = parse_tool_call(good)
    print(f"Good input:  {result}")
    assert result == ("query_transactions", {"category": "groceries", "month": "2026-01"})

    # Model invents a tool
    invented = '<tool_call name="send_money">{"to": "alice", "amount": 500}</tool_call>'
    result = parse_tool_call(invented)
    print(f"Invented fn: {result}")
    assert result == ("send_money", {"to": "alice", "amount": 500})  # Parser can't tell!

    # No tool call in output
    no_call = "Sure, I'd be happy to help you check your transactions!"
    result = parse_tool_call(no_call)
    print(f"No call:     {result}")
    assert result is None

    # Malformed JSON
    bad_json = '<tool_call name="query_transactions">{category: \'food\'}</tool_call>'
    result = parse_tool_call(bad_json)
    print(f"Bad JSON:    {result}")
    assert result is None

    # Tool call embedded in conversation
    embedded = 'Let me check that for you. <tool_call name="query_transactions">{"category": "groceries"}</tool_call> Here are your results.'
    result = parse_tool_call(embedded)
    print(f"Embedded:    {result}")
    assert result == ("query_transactions", {"category": "groceries"})

    print("\nAll 5 tests passed!")
