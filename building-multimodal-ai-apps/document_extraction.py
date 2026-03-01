"""
Document extraction with VLMs: single-image and multi-page PDF pipelines.

Blog post: https://dadops.dev/blog/building-multimodal-ai-apps/
Code Blocks 1 and 2.

Requires: openai, pydantic, PyMuPDF (fitz).
NOTE: Requires OPENAI_API_KEY environment variable to call GPT-4o.
"""
import base64
import json
from pathlib import Path
from dataclasses import dataclass, field

from pydantic import BaseModel


# ── Code Block 1: Structured Document Extraction ──

class LineItem(BaseModel):
    description: str
    quantity: float
    unit_price: float
    total: float


class InvoiceData(BaseModel):
    invoice_number: str
    date: str
    vendor_name: str
    total_amount: float
    line_items: list[LineItem]


def extract_document(image_path: str, schema: type[BaseModel]) -> BaseModel:
    """Extract structured data from a document image using a VLM."""
    from openai import OpenAI

    image_bytes = Path(image_path).read_bytes()
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    suffix = Path(image_path).suffix.lstrip(".")
    media_type = f"image/{'jpeg' if suffix == 'jpg' else suffix}"

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    f"Extract the following fields from this document image. "
                    f"Return valid JSON matching this schema:\n"
                    f"{json.dumps(schema.model_json_schema(), indent=2)}"
                )},
                {"type": "image_url", "image_url": {
                    "url": f"data:{media_type};base64,{b64_image}",
                    "detail": "high"
                }}
            ]
        }],
        max_tokens=2000
    )
    raw = json.loads(response.choices[0].message.content)
    return schema.model_validate(raw)


# ── Code Block 2: Multi-Page PDF Processing ──

@dataclass
class PageResult:
    page_number: int
    extracted: dict
    confidence: float


@dataclass
class DocumentResult:
    pages: list[PageResult] = field(default_factory=list)
    merged: dict = field(default_factory=dict)


def process_multipage_document(pdf_path: str, prompt: str) -> DocumentResult:
    """Process a multi-page PDF by converting each page to an image."""
    import fitz  # PyMuPDF
    from openai import OpenAI

    doc = fitz.open(pdf_path)
    result = DocumentResult()
    client = OpenAI()

    for page_num in range(len(doc)):
        # Render page at 200 DPI — good balance of quality vs token cost
        pix = doc[page_num].get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        b64_image = base64.b64encode(img_bytes).decode("utf-8")

        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        f"{prompt}\n\nThis is page {page_num + 1} of {len(doc)}. "
                        f"Also include a 'confidence' field (0-1) rating your "
                        f"extraction confidence."
                    )},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{b64_image}",
                        "detail": "high"
                    }}
                ]
            }],
            max_tokens=2000
        )
        page_data = json.loads(response.choices[0].message.content)
        confidence = page_data.pop("confidence", 0.8)
        result.pages.append(PageResult(page_num + 1, page_data, confidence))
        print(f"  Page {page_num + 1}/{len(doc)}: confidence {confidence:.0%}")

    doc.close()
    # Merge: combine line items, take the latest totals, flag low-confidence pages
    result.merged = merge_page_results(result.pages)
    return result


def merge_page_results(pages: list[PageResult]) -> dict:
    """Merge extracted data across pages, combining lists and flagging conflicts."""
    merged = {}
    for page in pages:
        for key, value in page.extracted.items():
            if isinstance(value, list):
                merged.setdefault(key, []).extend(value)
            else:
                merged[key] = value  # Later pages override earlier ones
    merged["low_confidence_pages"] = [
        p.page_number for p in pages if p.confidence < 0.7
    ]
    return merged


if __name__ == "__main__":
    # ── Self-tests for non-API components ──
    print("=== Document Extraction — Self Tests ===\n")

    # Test 1: Pydantic models validate correctly
    print("Test 1: Pydantic model validation...")
    invoice = InvoiceData.model_validate({
        "invoice_number": "INV-001",
        "date": "2026-01-15",
        "vendor_name": "Acme Corp",
        "total_amount": 250.00,
        "line_items": [
            {"description": "Widget A", "quantity": 5, "unit_price": 30.00, "total": 150.00},
            {"description": "Widget B", "quantity": 2, "unit_price": 50.00, "total": 100.00},
        ]
    })
    assert invoice.invoice_number == "INV-001"
    assert invoice.total_amount == 250.00
    assert len(invoice.line_items) == 2
    assert invoice.line_items[0].description == "Widget A"
    print(f"  InvoiceData validated: #{invoice.invoice_number}, ${invoice.total_amount:.2f}")
    print(f"  Line items: {len(invoice.line_items)}")
    print("  PASS\n")

    # Test 2: JSON schema generation
    print("Test 2: JSON schema generation...")
    schema = InvoiceData.model_json_schema()
    assert "invoice_number" in schema["properties"]
    assert "line_items" in schema["properties"]
    print(f"  Schema has {len(schema['properties'])} fields: {list(schema['properties'].keys())}")
    print("  PASS\n")

    # Test 3: merge_page_results
    print("Test 3: merge_page_results...")
    pages = [
        PageResult(1, {"vendor": "Acme", "items": ["A", "B"], "total": 100}, 0.9),
        PageResult(2, {"items": ["C", "D"], "total": 200}, 0.5),
        PageResult(3, {"items": ["E"], "notes": "final page"}, 0.8),
    ]
    merged = merge_page_results(pages)
    assert merged["items"] == ["A", "B", "C", "D", "E"], f"Got {merged['items']}"
    assert merged["total"] == 200  # Later page overrides
    assert merged["vendor"] == "Acme"
    assert merged["notes"] == "final page"
    assert merged["low_confidence_pages"] == [2]  # Page 2 had 0.5 confidence
    print(f"  Merged items: {merged['items']}")
    print(f"  Merged total: {merged['total']} (page 2 overrides page 1)")
    print(f"  Low confidence pages: {merged['low_confidence_pages']}")
    print("  PASS\n")

    print("All document extraction self-tests passed!")
    print("\nNote: API-dependent functions (extract_document, process_multipage_document)")
    print("require OPENAI_API_KEY to run.")
