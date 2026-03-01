"""
Alternative receipt extraction using Anthropic Claude.

From: https://dadops.dev/blog/llm-receipt-parser/

Claude Haiku is cheaper per receipt (~$0.0016) and nearly as accurate.
Requires ANTHROPIC_API_KEY environment variable.

Dependencies: anthropic
"""

import json
import anthropic

# Import the shared system prompt
from receipt_parser import SYSTEM_PROMPT


def extract_receipt_claude(image_b64):
    """Send receipt image to Claude Haiku and return parsed JSON."""
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4000,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_b64
                }},
                {"type": "text", "text": SYSTEM_PROMPT + "\n\nParse this receipt."}
            ]
        }]
    )
    return json.loads(response.content[0].text)


if __name__ == "__main__":
    print("=== Claude Receipt Parser ===")
    print("Requires ANTHROPIC_API_KEY environment variable.")
    print("Use: from receipt_parser_claude import extract_receipt_claude")
