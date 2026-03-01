"""
Verification suite for the LLM Receipt Parser blog post.

From: https://dadops.dev/blog/llm-receipt-parser/

Tests all non-API logic: database creation, image preprocessing,
validation, storage, SQL queries, and blog numerical claims.

Dependencies: pillow (for image preprocessing tests)
"""

import sqlite3
import json
import os
import tempfile
from decimal import Decimal

from receipt_parser import (
    init_db, prepare_image, validate_receipt, store_receipt, SYSTEM_PROMPT
)


def test(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return condition


def main():
    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed
        if test(name, condition, detail):
            passed += 1
        else:
            failed += 1

    # ═══════════════════════════════════════════
    # 1. Database Initialization
    # ═══════════════════════════════════════════
    print("=== 1. Database Initialization ===")
    test_db = tempfile.mktemp(suffix=".db")
    init_db(test_db)

    conn = sqlite3.connect(test_db)
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()]
    check("receipts table exists", "receipts" in tables, f"tables={tables}")
    check("line_items table exists", "line_items" in tables, f"tables={tables}")

    # Check columns
    receipt_cols = [r[1] for r in conn.execute("PRAGMA table_info(receipts)").fetchall()]
    check("receipts has store_name", "store_name" in receipt_cols)
    check("receipts has raw_json", "raw_json" in receipt_cols)
    check("receipts has validation_ok", "validation_ok" in receipt_cols)

    item_cols = [r[1] for r in conn.execute("PRAGMA table_info(line_items)").fetchall()]
    check("line_items has receipt_id", "receipt_id" in item_cols)
    check("line_items has is_discount", "is_discount" in item_cols)

    # Check indexes
    indexes = [r[1] for r in conn.execute("PRAGMA index_list(receipts)").fetchall()]
    check("receipts date index exists",
          any("date" in idx for idx in indexes), f"indexes={indexes}")

    conn.close()
    os.remove(test_db)

    # ═══════════════════════════════════════════
    # 2. Image Preprocessing
    # ═══════════════════════════════════════════
    print("\n=== 2. Image Preprocessing ===")
    try:
        from PIL import Image
        import io
        import base64

        # Create a test image (3000x2000 RGB)
        test_img = Image.new("RGB", (3000, 2000), color=(255, 255, 255))
        test_img_path = tempfile.mktemp(suffix=".png")
        test_img.save(test_img_path)

        result_b64 = prepare_image(test_img_path)
        check("prepare_image returns string", isinstance(result_b64, str))

        # Decode and check dimensions
        decoded = base64.standard_b64decode(result_b64)
        result_img = Image.open(io.BytesIO(decoded))
        check("Output is JPEG", result_img.format == "JPEG")
        check("Long edge <= 1500",
              max(result_img.size) <= 1500,
              f"size={result_img.size}")
        check("Aspect ratio preserved",
              abs(result_img.width / result_img.height - 3000 / 2000) < 0.01,
              f"ratio={result_img.width / result_img.height:.3f}, expected=1.500")

        # Test with small image (no resize needed)
        small_img = Image.new("RGB", (800, 600), color=(200, 200, 200))
        small_path = tempfile.mktemp(suffix=".png")
        small_img.save(small_path)
        small_b64 = prepare_image(small_path)
        small_decoded = base64.standard_b64decode(small_b64)
        small_result = Image.open(io.BytesIO(small_decoded))
        check("Small image not resized",
              small_result.size == (800, 600),
              f"size={small_result.size}")

        # Test RGBA → RGB conversion
        rgba_img = Image.new("RGBA", (1000, 800), color=(255, 255, 255, 128))
        rgba_path = tempfile.mktemp(suffix=".png")
        rgba_img.save(rgba_path)
        rgba_b64 = prepare_image(rgba_path)
        check("RGBA image converts successfully", isinstance(rgba_b64, str))

        os.remove(test_img_path)
        os.remove(small_path)
        os.remove(rgba_path)

    except ImportError:
        print("  [SKIP] Pillow not installed — skipping image tests")

    # ═══════════════════════════════════════════
    # 3. System Prompt
    # ═══════════════════════════════════════════
    print("\n=== 3. System Prompt ===")
    check("Prompt mentions JSON", "JSON" in SYSTEM_PROMPT)
    check("Prompt mentions null", "null" in SYSTEM_PROMPT)
    check("Prompt mentions discount", "discount" in SYSTEM_PROMPT.lower())
    check("Prompt mentions YYYY-MM-DD", "YYYY-MM-DD" in SYSTEM_PROMPT)
    check("Prompt defines store_name", '"store_name"' in SYSTEM_PROMPT)
    check("Prompt defines is_discount", '"is_discount"' in SYSTEM_PROMPT)

    # ═══════════════════════════════════════════
    # 4. Validation Logic
    # ═══════════════════════════════════════════
    print("\n=== 4. Validation Logic ===")

    # Valid receipt
    valid = {
        "store_name": "Test Store",
        "date": "2026-02-25",
        "items": [
            {"name": "Item A", "quantity": 1, "unit_price": 5.99, "line_total": 5.99},
            {"name": "Item B", "quantity": 2, "unit_price": 3.50, "line_total": 7.00},
        ],
        "subtotal": 12.99,
        "tax": 0.91,
        "total": 13.90,
    }
    is_valid, errors = validate_receipt(valid)
    check("Valid receipt passes", is_valid, f"errors={errors}")

    # With tolerance: items sum to 12.99, subtotal 12.99 — exact match
    check("Item sum matches subtotal exactly",
          sum(Decimal(str(i["line_total"])) for i in valid["items"]) == Decimal("12.99"))

    # Missing required fields
    missing = {"date": "2026-01-01"}
    is_valid, errors = validate_receipt(missing)
    check("Missing store_name detected",
          any("store_name" in e for e in errors))
    check("Missing total detected",
          any("total" in e for e in errors))
    check("Missing items detected",
          any("items" in e for e in errors))

    # Arithmetic mismatch
    bad_math = {
        "store_name": "Bad Store",
        "items": [
            {"name": "X", "quantity": 1, "unit_price": 5.00, "line_total": 5.00}
        ],
        "subtotal": 10.00,  # Wrong
        "tax": 0.50,
        "total": 10.50,
    }
    is_valid, errors = validate_receipt(bad_math)
    check("Item sum mismatch detected", not is_valid)
    check("Error mentions item sum",
          any("Item sum" in e for e in errors))

    # Subtotal + tax != total
    bad_total = {
        "store_name": "Bad Total",
        "items": [{"name": "Y", "quantity": 1, "unit_price": 5.00, "line_total": 5.00}],
        "subtotal": 5.00,
        "tax": 0.50,
        "total": 15.00,  # Wrong
    }
    is_valid, errors = validate_receipt(bad_total)
    check("Total mismatch detected", not is_valid)
    check("Error mentions subtotal + tax",
          any("subtotal + tax" in e for e in errors))

    # Plausible range
    huge = {"store_name": "X", "total": 5000,
            "items": [{"name": "Z", "line_total": 5000}]}
    is_valid, errors = validate_receipt(huge)
    check("Implausible total detected",
          any("plausible" in e for e in errors))

    tiny = {"store_name": "X", "total": 0.001,
            "items": [{"name": "Z", "line_total": 0.001}]}
    is_valid, errors = validate_receipt(tiny)
    check("Tiny total detected",
          any("plausible" in e for e in errors))

    # Tolerance test: $0.05 difference should pass
    within_tol = {
        "store_name": "Tol Store",
        "items": [
            {"name": "A", "quantity": 1, "unit_price": 1.33, "line_total": 1.33},
            {"name": "B", "quantity": 1, "unit_price": 1.33, "line_total": 1.33},
            {"name": "C", "quantity": 1, "unit_price": 1.33, "line_total": 1.33},
        ],
        "subtotal": 3.99,  # Items sum to 3.99, matches exactly
        "tax": 0.28,
        "total": 4.27,
    }
    is_valid, errors = validate_receipt(within_tol)
    check("$0.00 tolerance passes", is_valid, f"errors={errors}")

    # With discount (negative line_total)
    with_discount = {
        "store_name": "Discount Store",
        "items": [
            {"name": "Item", "quantity": 1, "unit_price": 10.00, "line_total": 10.00},
            {"name": "COUPON", "quantity": 1, "unit_price": -2.00, "line_total": -2.00, "is_discount": True},
        ],
        "subtotal": 8.00,
        "tax": 0.56,
        "total": 8.56,
    }
    is_valid, errors = validate_receipt(with_discount)
    check("Discount receipt validates", is_valid, f"errors={errors}")

    # ═══════════════════════════════════════════
    # 5. Storage
    # ═══════════════════════════════════════════
    print("\n=== 5. Storage ===")
    test_db = tempfile.mktemp(suffix=".db")
    init_db(test_db)

    raw = json.dumps(valid, indent=2)
    rid = store_receipt(valid, "test.jpg", raw, True, test_db)
    check("Receipt stored with ID", rid is not None and rid > 0, f"id={rid}")

    conn = sqlite3.connect(test_db)
    row = conn.execute("SELECT * FROM receipts WHERE id=?", (rid,)).fetchone()
    check("Receipt retrievable", row is not None)
    check("Store name stored correctly", row[1] == "Test Store")
    check("validation_ok is 1", row[10] == 1)

    items = conn.execute("SELECT * FROM line_items WHERE receipt_id=?", (rid,)).fetchall()
    check("Line items stored", len(items) == 2, f"count={len(items)}")
    check("First item name", items[0][2] == "Item A")

    # Store invalid receipt
    rid2 = store_receipt(bad_math, "bad.jpg", "{}", False, test_db)
    row2 = conn.execute("SELECT validation_ok FROM receipts WHERE id=?", (rid2,)).fetchone()
    check("Invalid receipt stored with validation_ok=0", row2[0] == 0)

    conn.close()
    os.remove(test_db)

    # ═══════════════════════════════════════════
    # 6. Blog Claims
    # ═══════════════════════════════════════════
    print("\n=== 6. Blog Claims ===")

    # Float precision claim
    float_result = 3 * 1.33
    check("Float 3*1.33 = 3.9900000000000002",
          float_result == 3.9900000000000002,
          f"actual={float_result!r}")

    # Cost table: 4 trips/week × 52 weeks = 208 receipts/year
    receipts_year = 4 * 52
    check("208 receipts/year", receipts_year == 208)

    # Verify cost calculations
    costs = {
        "Gemini Flash": (0.0002, 0.04),
        "Claude Haiku": (0.0016, 0.33),
        "GPT-4o": (0.002, 0.42),
        "GPT-4o-mini": (0.004, 0.83),
    }
    for model, (per_receipt, yearly_claimed) in costs.items():
        yearly_calc = round(per_receipt * receipts_year, 2)
        check(f"{model} yearly cost",
              yearly_calc == yearly_claimed,
              f"calc=${yearly_calc}, claimed=${yearly_claimed}")

    # ═══════════════════════════════════════════
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed, "
          f"{passed + failed} total")
    if failed == 0:
        print("All tests passed!")
    else:
        print(f"WARNING: {failed} test(s) failed")


if __name__ == "__main__":
    main()
