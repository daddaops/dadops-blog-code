# LLM Receipt Parser — Verified Code

Runnable code from [Using LLMs to Parse Grocery Receipts](https://daddaops.com/blog/llm-receipt-parser/) on DadOps.

## Scripts

| Script | Description | API Key? |
|--------|-------------|----------|
| `receipt_parser.py` | Main module: init_db, prepare_image, extract_receipt, validate_receipt, store_receipt, parse_receipt | OpenAI (for extraction only) |
| `receipt_parser_claude.py` | Alternative extraction using Anthropic Claude | Anthropic |
| `validate_demo.py` | Standalone demo of validation + SQLite logic with mock data | No |
| `verify_receipt_parser.py` | Full verification suite testing all non-API logic | No (uses Pillow) |
| `queries.sql` | SQL analysis queries from the blog post | No |

## Quick Start

```bash
pip install -r requirements.txt
python3 validate_demo.py        # Test validation and SQLite (no API key)
python3 verify_receipt_parser.py # Run full test suite
```

## Full Pipeline (requires OpenAI API key)

```bash
export OPENAI_API_KEY="your-key-here"
python3 -c "from receipt_parser import init_db, parse_receipt; init_db(); parse_receipt('receipt.jpg')"
```
