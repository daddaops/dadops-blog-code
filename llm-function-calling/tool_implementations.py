"""
Shared tool implementations used by both OpenAI and Anthropic agent loops.

From: https://dadops.dev/blog/llm-function-calling/

These are the three mock tools for the personal finance assistant demo:
- query_transactions: Search SQLite for transaction data (simplified)
- convert_currency: Convert between currencies using hardcoded rates
- calculate: Safely evaluate math expressions

No API key required — these are pure functions with mock data.
"""

import json


def query_transactions(category=None, month=None, account=None):
    """Query SQLite for transaction data. (Simplified for demo.)"""
    if account and not category:
        return {"balance": 4250.00, "currency": "USD", "as_of": "2026-02-26"}
    return {"total": 847.32, "currency": "USD", "count": 23, "category": category}


def convert_currency(amount, from_currency, to_currency):
    """Convert between currencies using current rates."""
    rates = {"USD_EUR": 0.9231, "USD_GBP": 0.7891, "EUR_USD": 1.0833}
    rate = rates.get(f"{from_currency}_{to_currency}")
    if not rate:
        raise ValueError(f"Unknown currency pair: {from_currency}/{to_currency}")
    return {"converted": round(amount * rate, 2), "rate": rate}


def calculate(expression):
    """Safely evaluate a math expression."""
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        raise ValueError(f"Unsafe expression: {expression}")
    return {"result": round(eval(expression), 2)}


TOOLS = {
    "query_transactions": query_transactions,
    "convert_currency": convert_currency,
    "calculate": calculate,
}

# OpenAI tool schema format
OPENAI_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "query_transactions",
            "description": "Search transaction history by category, month, or account",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "e.g. groceries, restaurants"},
                    "month": {"type": "string", "description": "YYYY-MM format"},
                    "account": {"type": "string", "enum": ["checking", "savings", "all"]},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "convert_currency",
            "description": "Convert an amount between currencies",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number"},
                    "from_currency": {"type": "string", "description": "3-letter code like USD"},
                    "to_currency": {"type": "string", "description": "3-letter code like EUR"},
                },
                "required": ["amount", "from_currency", "to_currency"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a math expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression like '847.32 * 0.9231'"},
                },
                "required": ["expression"],
            },
        },
    },
]

# Anthropic tool schema format
ANTHROPIC_TOOLS_SCHEMA = [
    {
        "name": "query_transactions",
        "description": "Search transaction history by category, month, or account",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "e.g. groceries, restaurants"},
                "month": {"type": "string", "description": "YYYY-MM format"},
                "account": {"type": "string", "enum": ["checking", "savings", "all"]},
            },
            "required": [],
        },
    },
    {
        "name": "convert_currency",
        "description": "Convert an amount between currencies",
        "input_schema": {
            "type": "object",
            "properties": {
                "amount": {"type": "number"},
                "from_currency": {"type": "string", "description": "3-letter code like USD"},
                "to_currency": {"type": "string", "description": "3-letter code like EUR"},
            },
            "required": ["amount", "from_currency", "to_currency"],
        },
    },
    {
        "name": "calculate",
        "description": "Evaluate a math expression",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression like '847.32 * 0.9231'"},
            },
            "required": ["expression"],
        },
    },
]


if __name__ == "__main__":
    print("=== Tool Implementation Tests ===\n")

    # query_transactions tests
    r = query_transactions(category="groceries", month="2026-01")
    print(f"Transactions (category): {r}")
    assert r == {"total": 847.32, "currency": "USD", "count": 23, "category": "groceries"}

    r = query_transactions(account="checking")
    print(f"Transactions (balance):  {r}")
    assert r == {"balance": 4250.00, "currency": "USD", "as_of": "2026-02-26"}

    # convert_currency tests
    r = convert_currency(100, "USD", "EUR")
    print(f"100 USD → EUR:           {r}")
    assert r == {"converted": 92.31, "rate": 0.9231}

    r = convert_currency(847.32, "USD", "EUR")
    print(f"847.32 USD → EUR:        {r}")
    assert r == {"converted": round(847.32 * 0.9231, 2), "rate": 0.9231}

    try:
        convert_currency(100, "USD", "JPY")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"Unknown pair error:      {e}")

    # calculate tests
    r = calculate("847.32 * 0.9231")
    print(f"847.32 * 0.9231:         {r}")
    assert r == {"result": round(847.32 * 0.9231, 2)}

    r = calculate("(100 + 200) / 3")
    print(f"(100 + 200) / 3:         {r}")
    assert r == {"result": 100.0}

    try:
        calculate("__import__('os').system('rm -rf /')")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"Unsafe expression:       {e}")

    # Schema validation
    assert len(OPENAI_TOOLS_SCHEMA) == 3
    assert len(ANTHROPIC_TOOLS_SCHEMA) == 3
    assert OPENAI_TOOLS_SCHEMA[0]["type"] == "function"
    assert "input_schema" in ANTHROPIC_TOOLS_SCHEMA[0]
    print(f"\nOpenAI schema:           {len(OPENAI_TOOLS_SCHEMA)} tools defined")
    print(f"Anthropic schema:        {len(ANTHROPIC_TOOLS_SCHEMA)} tools defined")

    print("\nAll tests passed!")
