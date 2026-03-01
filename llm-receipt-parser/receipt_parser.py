"""
Main receipt parser module — combines all code blocks from the blog post.

From: https://dadops.dev/blog/llm-receipt-parser/

Pipeline: Receipt Photo → Resize & Encode → Vision LLM → Validate JSON → SQLite

Code Blocks 1-6 combined: init_db, prepare_image, SYSTEM_PROMPT,
extract_receipt (OpenAI), validate_receipt, store_receipt, parse_receipt.

Dependencies: openai, pillow (only openai needed for actual API calls)
"""

import sqlite3
import json
import base64
import io
from decimal import Decimal
from PIL import Image


# ═══════════════════════════════════════════
# Code Block 1: Database Setup
# ═══════════════════════════════════════════

def init_db(db_path="receipts.db"):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS receipts (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            store_name    TEXT NOT NULL,
            receipt_date  TEXT,
            subtotal      REAL,
            tax           REAL,
            total         REAL,
            payment_method TEXT,
            image_path    TEXT,
            raw_json      TEXT,
            parsed_at     TEXT DEFAULT (datetime('now')),
            validation_ok INTEGER DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS line_items (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            receipt_id    INTEGER NOT NULL
                          REFERENCES receipts(id) ON DELETE CASCADE,
            name          TEXT NOT NULL,
            quantity      REAL DEFAULT 1.0,
            unit_price    REAL,
            line_total    REAL,
            is_discount   INTEGER DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_receipts_date
            ON receipts(receipt_date);
        CREATE INDEX IF NOT EXISTS idx_items_receipt
            ON line_items(receipt_id);
    """)
    conn.close()


# ═══════════════════════════════════════════
# Code Block 2: Image Preprocessing
# ═══════════════════════════════════════════

def prepare_image(image_path):
    """Resize receipt image and return base64-encoded JPEG."""
    with Image.open(image_path) as img:
        # Convert RGBA/palette images to RGB for JPEG
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        # Resize: keep long edge under 1500px
        max_dim = 1500
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)

        # Encode as JPEG (smaller than PNG, fine for text)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")


# ═══════════════════════════════════════════
# Code Block 3: System Prompt
# ═══════════════════════════════════════════

SYSTEM_PROMPT = """You are a receipt parser. Extract all data from the receipt image
and return it as a single JSON object matching this exact structure:

{
  "store_name": "string",
  "date": "YYYY-MM-DD or null",
  "items": [
    {
      "name": "string",
      "quantity": number,
      "unit_price": number,
      "line_total": number,
      "is_discount": false
    }
  ],
  "subtotal": number,
  "tax": number,
  "total": number,
  "payment_method": "cash|credit|debit|unknown"
}

Rules:
- Return ONLY valid JSON. No markdown, no explanation.
- Use null for any field you cannot read. Do not guess or invent data.
- For weight-based items (e.g., "0.73 lb @ $2.99/lb"), use the per-unit
  price as unit_price and the extended price as line_total.
- For discounts and coupons, create an item entry with is_discount: true
  and a NEGATIVE line_total.
- Dates must be YYYY-MM-DD format. Convert from any format on the receipt.
- Item names should be the full text as printed (preserve abbreviations).
- If the receipt shows multiple tax lines, sum them into a single tax value."""


# ═══════════════════════════════════════════
# Code Block 4: OpenAI API Call (requires API key)
# ═══════════════════════════════════════════

def extract_receipt(image_b64):
    """Send receipt image to GPT-4o and return parsed JSON."""
    from openai import OpenAI
    client = OpenAI()  # reads OPENAI_API_KEY from environment
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        max_tokens=4000,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}",
                        "detail": "high"
                    }
                },
                {"type": "text", "text": "Parse this receipt."}
            ]}
        ]
    )
    return json.loads(response.choices[0].message.content)


# ═══════════════════════════════════════════
# Code Block 5: Validation
# ═══════════════════════════════════════════

def validate_receipt(data):
    """Check that receipt numbers add up. Returns (is_valid, errors)."""
    errors = []

    # Check required fields
    for field in ("store_name", "total", "items"):
        if not data.get(field):
            errors.append(f"Missing required field: {field}")

    items = data.get("items", [])
    if not items:
        return len(errors) == 0, errors

    # Do line items sum to subtotal?
    if data.get("subtotal"):
        item_sum = sum(
            Decimal(str(item["line_total"]))
            for item in items
        )
        subtotal = Decimal(str(data["subtotal"]))
        if abs(item_sum - subtotal) > Decimal("0.10"):
            errors.append(
                f"Item sum ${item_sum} != subtotal ${subtotal}"
            )

    # Does subtotal + tax = total?
    if all(data.get(k) is not None for k in ("subtotal", "tax", "total")):
        calculated = (
            Decimal(str(data["subtotal"]))
            + Decimal(str(data["tax"]))
        )
        stated = Decimal(str(data["total"]))
        if abs(calculated - stated) > Decimal("0.10"):
            errors.append(
                f"subtotal + tax = ${calculated}, but total = ${stated}"
            )

    # Sanity check
    total = data.get("total", 0)
    if total and (total < 0.01 or total > 2000):
        errors.append(f"Total ${total} outside plausible range")

    return len(errors) == 0, errors


# ═══════════════════════════════════════════
# Code Block 6: Storage and Pipeline
# ═══════════════════════════════════════════

def store_receipt(data, image_path, raw_json, is_valid, db_path="receipts.db"):
    """Insert parsed receipt and line items into SQLite."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        cursor = conn.execute("""
            INSERT INTO receipts
                (store_name, receipt_date, subtotal, tax, total,
                 payment_method, image_path, raw_json, validation_ok)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data.get("store_name", "Unknown"),
            data.get("date"),
            data.get("subtotal"),
            data.get("tax"),
            data.get("total"),
            data.get("payment_method"),
            image_path,
            raw_json,
            1 if is_valid else 0
        ))
        receipt_id = cursor.lastrowid

        for item in data.get("items", []):
            conn.execute("""
                INSERT INTO line_items
                    (receipt_id, name, quantity, unit_price,
                     line_total, is_discount)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                receipt_id,
                item["name"],
                item.get("quantity", 1.0),
                item.get("unit_price"),
                item.get("line_total"),
                1 if item.get("is_discount") else 0
            ))

        conn.commit()
        return receipt_id
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def parse_receipt(image_path, db_path="receipts.db"):
    """Full pipeline: image → LLM → validate → store."""
    # 1. Preprocess
    image_b64 = prepare_image(image_path)

    # 2. Extract
    data = extract_receipt(image_b64)
    raw_json = json.dumps(data, indent=2)

    # 3. Validate
    is_valid, errors = validate_receipt(data)
    if errors:
        print(f"Validation warnings: {errors}")

    # 4. Store
    receipt_id = store_receipt(data, image_path, raw_json, is_valid, db_path)
    print(f"Receipt #{receipt_id}: {data.get('store_name')} "
          f"${data.get('total')} ({'OK' if is_valid else 'NEEDS REVIEW'})")
    return receipt_id


if __name__ == "__main__":
    print("=== Receipt Parser Module ===\n")
    print("This module requires an OpenAI API key to run the full pipeline.")
    print("Use validate_demo.py and verify_receipt_parser.py to test")
    print("the validation, SQLite, and image preprocessing logic.")
