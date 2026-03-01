"""
Standalone demo of the validation and SQLite logic — no API keys needed.

From: https://dadops.dev/blog/llm-receipt-parser/

Tests validate_receipt() with mock receipt data and runs all four
SQL analysis queries against a populated test database.

No external dependencies required (uses only stdlib + receipt_parser module).
"""

import sqlite3
import json
import os
from decimal import Decimal

from receipt_parser import init_db, validate_receipt, store_receipt


def main():
    print("=== Receipt Parser Validation Demo ===\n")

    # ── Test 1: Valid receipt ──────────────────────────
    valid_receipt = {
        "store_name": "Costco Wholesale",
        "date": "2026-02-20",
        "items": [
            {"name": "KS ORGANIC MILK 2PK", "quantity": 1, "unit_price": 9.99, "line_total": 9.99, "is_discount": False},
            {"name": "BANANAS", "quantity": 1, "unit_price": 1.49, "line_total": 1.49, "is_discount": False},
            {"name": "KS ROTISSERIE CHICKEN", "quantity": 1, "unit_price": 4.99, "line_total": 4.99, "is_discount": False},
            {"name": "STRAWBERRIES 2LB", "quantity": 2, "unit_price": 5.99, "line_total": 11.98, "is_discount": False},
            {"name": "INSTANT SAVINGS", "quantity": 1, "unit_price": -3.00, "line_total": -3.00, "is_discount": True},
        ],
        "subtotal": 25.45,
        "tax": 1.87,
        "total": 27.32,
        "payment_method": "credit"
    }

    is_valid, errors = validate_receipt(valid_receipt)
    print(f"Test 1 - Valid receipt: valid={is_valid}, errors={errors}")
    assert is_valid, f"Expected valid, got errors: {errors}"
    print("  ✓ PASS\n")

    # Verify arithmetic: 9.99+1.49+4.99+11.98+(-3.00) = 25.45
    item_sum = sum(Decimal(str(i["line_total"])) for i in valid_receipt["items"])
    print(f"  Item sum: {item_sum} (expected 25.45)")
    print(f"  Subtotal + tax: {Decimal('25.45') + Decimal('1.87')} (expected 27.32)")

    # ── Test 2: Invalid receipt (math doesn't add up) ─
    print()
    bad_receipt = {
        "store_name": "Trader Joe's",
        "date": "2026-02-21",
        "items": [
            {"name": "ORANGE JUICE", "quantity": 1, "unit_price": 3.99, "line_total": 3.99, "is_discount": False},
            {"name": "FROZEN PIZZA", "quantity": 2, "unit_price": 4.49, "line_total": 8.98, "is_discount": False},
        ],
        "subtotal": 15.00,  # Wrong! Should be 12.97
        "tax": 0.95,
        "total": 15.95,
        "payment_method": "debit"
    }

    is_valid, errors = validate_receipt(bad_receipt)
    print(f"Test 2 - Bad arithmetic: valid={is_valid}, errors={errors}")
    assert not is_valid, "Expected invalid"
    assert any("Item sum" in e for e in errors), "Expected item sum error"
    print("  ✓ PASS\n")

    # ── Test 3: Missing required fields ───────────────
    incomplete = {"date": "2026-02-22", "total": 10.00}
    is_valid, errors = validate_receipt(incomplete)
    print(f"Test 3 - Missing fields: valid={is_valid}, errors={errors}")
    assert not is_valid
    assert any("store_name" in e for e in errors)
    assert any("items" in e for e in errors)
    print("  ✓ PASS\n")

    # ── Test 4: Total outside plausible range ─────────
    huge = {
        "store_name": "Walmart", "total": 5000, "items": [
            {"name": "TEST", "quantity": 1, "unit_price": 5000, "line_total": 5000}
        ]
    }
    is_valid, errors = validate_receipt(huge)
    print(f"Test 4 - Implausible total: valid={is_valid}, errors={errors}")
    assert not is_valid
    assert any("plausible" in e for e in errors)
    print("  ✓ PASS\n")

    # ── Blog claim: float vs Decimal ──────────────────
    print("=== Blog Claim: Float Precision ===")
    float_result = 3 * 1.33
    decimal_result = Decimal("3") * Decimal("1.33")
    print(f"  float:   3 * 1.33 = {float_result!r}")
    print(f"  Decimal: 3 * 1.33 = {decimal_result}")
    print(f"  Blog claims float gives 3.9900000000000002")
    print(f"  Actual: {float_result == 3.9900000000000002}")
    print()

    # ── Test 5: SQLite integration ────────────────────
    print("=== SQLite Integration ===\n")
    test_db = "/tmp/test_receipts.db"
    if os.path.exists(test_db):
        os.remove(test_db)

    init_db(test_db)

    # Store valid receipt
    raw = json.dumps(valid_receipt, indent=2)
    receipt_id = store_receipt(valid_receipt, "costco_2026-02-20.jpg", raw, True, test_db)
    print(f"  Stored receipt #{receipt_id}")

    # Store a second receipt for query testing
    receipt2 = {
        "store_name": "Safeway",
        "date": "2026-02-22",
        "items": [
            {"name": "WHOLE MILK 1GAL", "quantity": 1, "unit_price": 4.79, "line_total": 4.79, "is_discount": False},
            {"name": "EGGS LARGE 12CT", "quantity": 1, "unit_price": 3.29, "line_total": 3.29, "is_discount": False},
            {"name": "BANANAS", "quantity": 1, "unit_price": 0.69, "line_total": 0.69, "is_discount": False},
        ],
        "subtotal": 8.77,
        "tax": 0.34,
        "total": 9.11,
        "payment_method": "credit"
    }
    raw2 = json.dumps(receipt2, indent=2)
    receipt_id2 = store_receipt(receipt2, "safeway_2026-02-22.jpg", raw2, True, test_db)
    print(f"  Stored receipt #{receipt_id2}")

    # ── Run the blog's SQL queries ────────────────────
    conn = sqlite3.connect(test_db)
    conn.row_factory = sqlite3.Row

    print("\n--- Monthly Spending ---")
    rows = conn.execute("""
        SELECT
            strftime('%Y-%m', receipt_date) AS month,
            COUNT(*) AS trips,
            ROUND(SUM(total), 2) AS spent
        FROM receipts
        WHERE receipt_date IS NOT NULL
        GROUP BY month
        ORDER BY month DESC;
    """).fetchall()
    for row in rows:
        print(f"  {row['month']}: {row['trips']} trips, ${row['spent']}")

    print("\n--- Most Purchased Items ---")
    rows = conn.execute("""
        SELECT
            name,
            COUNT(*) AS times_bought,
            ROUND(AVG(unit_price), 2) AS avg_price,
            ROUND(MIN(unit_price), 2) AS cheapest,
            ROUND(MAX(unit_price), 2) AS priciest
        FROM line_items
        WHERE is_discount = 0
        GROUP BY name
        HAVING times_bought > 1
        ORDER BY times_bought DESC
        LIMIT 15;
    """).fetchall()
    for row in rows:
        print(f"  {row['name']}: bought {row['times_bought']}x, avg ${row['avg_price']}")

    print("\n--- Price Tracking (BANANAS) ---")
    rows = conn.execute("""
        SELECT
            r.receipt_date,
            r.store_name,
            li.unit_price
        FROM line_items li
        JOIN receipts r ON li.receipt_id = r.id
        WHERE li.name LIKE '%BANANA%'
        ORDER BY r.receipt_date;
    """).fetchall()
    for row in rows:
        print(f"  {row['receipt_date']} @ {row['store_name']}: ${row['unit_price']}")

    print("\n--- Cheapest Store for BANANAS ---")
    rows = conn.execute("""
        SELECT
            r.store_name,
            ROUND(AVG(li.unit_price), 2) AS avg_price,
            COUNT(*) AS samples
        FROM line_items li
        JOIN receipts r ON li.receipt_id = r.id
        WHERE li.name LIKE '%BANANA%'
        GROUP BY r.store_name
        ORDER BY avg_price;
    """).fetchall()
    for row in rows:
        print(f"  {row['store_name']}: avg ${row['avg_price']} ({row['samples']} samples)")

    conn.close()
    os.remove(test_db)
    print(f"\n  Cleaned up {test_db}")

    # ── Cost verification ─────────────────────────────
    print("\n=== Cost Verification ===")
    print("Blog cost table claims:")
    costs = [
        ("Gemini Flash", 0.0002, 0.04),
        ("Claude Haiku", 0.0016, 0.33),
        ("GPT-4o", 0.002, 0.42),
        ("GPT-4o-mini", 0.004, 0.83),
    ]
    receipts_per_year = 4 * 52  # 4 trips/week × 52 weeks
    print(f"  Receipts/year: {receipts_per_year}")
    for model, per_receipt, yearly_claimed in costs:
        yearly_calc = round(per_receipt * receipts_per_year, 2)
        match = "✓" if yearly_calc == yearly_claimed else "✗"
        print(f"  {model}: ${per_receipt}/receipt × {receipts_per_year} = "
              f"${yearly_calc} (blog: ${yearly_claimed}) {match}")

    print("\nAll tests complete.")


if __name__ == "__main__":
    main()
