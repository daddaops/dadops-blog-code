"""
Semantic validation for structured LLM output.

Demonstrates a Receipt model with:
- Field validators (store name not empty, prices >= 0)
- Model validators (total matches subtotal + tax, line items sum to subtotal)

The Pydantic model runs without any API key — the validation logic
is tested with synthetic data.

From: https://dadops.dev/blog/structured-output-from-llms/
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List


class LineItem(BaseModel):
    description: str
    price: float = Field(ge=0)


class Receipt(BaseModel):
    store_name: str
    items: List[LineItem]
    subtotal: float = Field(ge=0)
    tax: float = Field(ge=0)
    total: float = Field(ge=0)

    @field_validator("store_name")
    @classmethod
    def store_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Store name cannot be empty")
        return v.strip()

    @model_validator(mode="after")
    def check_total_matches(self):
        """Semantic validation: total should equal subtotal + tax"""
        expected = round(self.subtotal + self.tax, 2)
        if abs(self.total - expected) > 0.02:  # 2 cent tolerance
            raise ValueError(
                f"Total ${self.total:.2f} doesn't match "
                f"subtotal ${self.subtotal:.2f} + tax ${self.tax:.2f} = ${expected:.2f}"
            )
        return self

    @model_validator(mode="after")
    def check_items_sum(self):
        """Semantic validation: line items should sum to subtotal"""
        items_sum = round(sum(item.price for item in self.items), 2)
        if abs(items_sum - self.subtotal) > 0.05:  # 5 cent tolerance
            raise ValueError(
                f"Line items sum to ${items_sum:.2f} "
                f"but subtotal is ${self.subtotal:.2f}"
            )
        return self


if __name__ == "__main__":
    print("=== Semantic Validation Demo ===\n")

    # Valid receipt
    print("1. Valid receipt:")
    try:
        receipt = Receipt(
            store_name="Trader Joe's",
            items=[
                LineItem(description="Bananas", price=1.29),
                LineItem(description="Coffee", price=8.99),
                LineItem(description="Bread", price=3.49),
            ],
            subtotal=13.77,
            tax=1.10,
            total=14.87,
        )
        print(f"   PASS: {receipt.store_name}, total=${receipt.total:.2f}")
    except Exception as e:
        print(f"   FAIL: {e}")

    # Bad total (doesn't match subtotal + tax)
    print("\n2. Bad total (subtotal + tax != total):")
    try:
        receipt = Receipt(
            store_name="Whole Foods",
            items=[LineItem(description="Avocado", price=2.50)],
            subtotal=2.50,
            tax=0.20,
            total=45.00,  # wrong!
        )
        print(f"   Should not reach here: {receipt}")
    except Exception as e:
        print(f"   CAUGHT: {e}")

    # Bad items sum (line items don't add to subtotal)
    print("\n3. Bad items sum (items don't add to subtotal):")
    try:
        receipt = Receipt(
            store_name="Target",
            items=[
                LineItem(description="Shirt", price=15.00),
                LineItem(description="Socks", price=5.00),
            ],
            subtotal=42.50,  # should be 20.00
            tax=3.40,
            total=45.90,
        )
        print(f"   Should not reach here: {receipt}")
    except Exception as e:
        print(f"   CAUGHT: {e}")

    # Empty store name
    print("\n4. Empty store name:")
    try:
        receipt = Receipt(
            store_name="  ",
            items=[LineItem(description="Widget", price=10.00)],
            subtotal=10.00,
            tax=0.80,
            total=10.80,
        )
        print(f"   Should not reach here: {receipt}")
    except Exception as e:
        print(f"   CAUGHT: {e}")
