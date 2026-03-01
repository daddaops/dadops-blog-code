"""
Verification script for LLM Function Calling blog post.

Runs all testable code without requiring API keys:
- Regex-based tool call parser
- Tool implementations (query_transactions, convert_currency, calculate)
- Schema validation (OpenAI and Anthropic formats)
"""

import json
from parse_tool_call import parse_tool_call, SYSTEM_PROMPT
from tool_implementations import (
    query_transactions, convert_currency, calculate,
    TOOLS, OPENAI_TOOLS_SCHEMA, ANTHROPIC_TOOLS_SCHEMA,
)

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
print("LLM Function Calling — Verification Suite")
print("=" * 60)

# --- Parser Tests ---
print("\n--- Regex Tool Call Parser ---")

result = parse_tool_call('<tool_call name="query_transactions">{"category": "groceries"}</tool_call>')
check("Valid tool call parsed", result == ("query_transactions", {"category": "groceries"}))

result = parse_tool_call("No tool call here, just text.")
check("No tool call returns None", result is None)

result = parse_tool_call('<tool_call name="query_transactions">{bad json}</tool_call>')
check("Malformed JSON returns None", result is None)

result = parse_tool_call('Sure! <tool_call name="calc">{"expr": "1+1"}</tool_call> Done.')
check("Embedded tool call extracted", result == ("calc", {"expr": "1+1"}))

result = parse_tool_call('<tool_call name="send_money">{"to": "alice"}</tool_call>')
check("Hallucinated fn name still parses", result is not None and result[0] == "send_money")

check("System prompt contains tool description", "query_transactions" in SYSTEM_PROMPT)
check("System prompt contains XML format", "<tool_call" in SYSTEM_PROMPT)

# --- Tool Implementation Tests ---
print("\n--- query_transactions ---")

r = query_transactions(category="groceries", month="2026-01")
check("Returns total for category query", r["total"] == 847.32)
check("Returns count for category query", r["count"] == 23)
check("Returns currency USD", r["currency"] == "USD")

r = query_transactions(account="checking")
check("Returns balance for account query", r["balance"] == 4250.00)
check("Returns as_of date", "as_of" in r)

print("\n--- convert_currency ---")

r = convert_currency(100, "USD", "EUR")
check("100 USD → EUR = 92.31", r["converted"] == 92.31)
check("USD→EUR rate = 0.9231", r["rate"] == 0.9231)

r = convert_currency(847.32, "USD", "EUR")
expected = round(847.32 * 0.9231, 2)
check(f"847.32 USD → EUR = {expected}", r["converted"] == expected)

r = convert_currency(100, "USD", "GBP")
check("100 USD → GBP = 78.91", r["converted"] == 78.91)

try:
    convert_currency(100, "USD", "JPY")
    check("Unknown pair raises error", False)
except ValueError:
    check("Unknown pair raises ValueError", True)

print("\n--- calculate ---")

r = calculate("847.32 * 0.9231")
check(f"847.32 * 0.9231 = {round(847.32 * 0.9231, 2)}", r["result"] == round(847.32 * 0.9231, 2))

r = calculate("(100 + 200) / 3")
check("(100 + 200) / 3 = 100.0", r["result"] == 100.0)

r = calculate("2 + 2")
check("2 + 2 = 4", r["result"] == 4)

try:
    calculate("__import__('os').system('ls')")
    check("Unsafe expression blocked", False)
except ValueError:
    check("Unsafe expression raises ValueError", True)

try:
    calculate("import os")
    check("Import blocked", False)
except ValueError:
    check("Import attempt raises ValueError", True)

# --- Tool Registry Tests ---
print("\n--- Tool Registry ---")

check("TOOLS has 3 entries", len(TOOLS) == 3)
check("query_transactions in TOOLS", "query_transactions" in TOOLS)
check("convert_currency in TOOLS", "convert_currency" in TOOLS)
check("calculate in TOOLS", "calculate" in TOOLS)

# --- Schema Validation ---
print("\n--- OpenAI Schema ---")

check("OpenAI schema has 3 tools", len(OPENAI_TOOLS_SCHEMA) == 3)
for tool in OPENAI_TOOLS_SCHEMA:
    check(f"  {tool['function']['name']} has type='function'", tool["type"] == "function")
    check(f"  {tool['function']['name']} has parameters", "parameters" in tool["function"])

print("\n--- Anthropic Schema ---")

check("Anthropic schema has 3 tools", len(ANTHROPIC_TOOLS_SCHEMA) == 3)
for tool in ANTHROPIC_TOOLS_SCHEMA:
    check(f"  {tool['name']} has input_schema", "input_schema" in tool)
    check(f"  {tool['name']} has no 'type' wrapper", "type" not in tool)

# --- Cross-schema Consistency ---
print("\n--- Cross-Schema Consistency ---")

for i in range(3):
    oai_name = OPENAI_TOOLS_SCHEMA[i]["function"]["name"]
    ant_name = ANTHROPIC_TOOLS_SCHEMA[i]["name"]
    check(f"Tool {i} name matches: {oai_name}", oai_name == ant_name)

    oai_desc = OPENAI_TOOLS_SCHEMA[i]["function"]["description"]
    ant_desc = ANTHROPIC_TOOLS_SCHEMA[i]["description"]
    check(f"Tool {i} description matches", oai_desc == ant_desc)

# --- Summary ---
print("\n" + "=" * 60)
print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
print("=" * 60)

if failed > 0:
    exit(1)
