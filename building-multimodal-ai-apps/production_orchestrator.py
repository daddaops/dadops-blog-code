"""
Production VLM orchestrator: preprocessing, token estimation, fallback chains, cost tracking.

Blog post: https://dadops.dev/blog/building-multimodal-ai-apps/
Code Block 7.

The image preprocessing and token estimation functions run without API keys.
The full fallback pipeline requires OPENAI_API_KEY and/or ANTHROPIC_API_KEY.

Requires: Pillow.
"""
import base64
import io
from dataclasses import dataclass
from PIL import Image


# ── Code Block 7: Production VLM Orchestrator ──

@dataclass
class VLMResult:
    data: dict
    model_used: str
    image_tokens: int
    cost_usd: float
    confidence: float


# Token costs per million (Feb 2026 pricing)
PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "claude-sonnet": {"input": 3.00, "output": 15.00},
    "gemini-pro": {"input": 1.25, "output": 5.00},
}


def preprocess_image(image_path: str, max_dimension: int = 2048) -> bytes:
    """Resize image to optimal dimensions for VLM processing."""
    img = Image.open(image_path)
    width, height = img.size

    # Skip resize if already within bounds
    if max(width, height) <= max_dimension:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    # Scale down preserving aspect ratio
    scale = max_dimension / max(width, height)
    new_size = (int(width * scale), int(height * scale))
    img = img.resize(new_size, Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def estimate_image_tokens_gpt4o(width: int, height: int, detail: str) -> int:
    """Estimate token count for an image sent to GPT-4o."""
    if detail == "low":
        return 85
    # High detail: scale long side to 2048, then short side to 768, then tile 512x512
    scale = min(2048 / max(width, height), 1.0)
    scaled_w, scaled_h = int(width * scale), int(height * scale)
    short_scale = min(768 / min(scaled_w, scaled_h), 1.0)
    scaled_w, scaled_h = int(scaled_w * short_scale), int(scaled_h * short_scale)
    tiles_w = (scaled_w + 511) // 512
    tiles_h = (scaled_h + 511) // 512
    return 85 + (tiles_w * tiles_h * 170)


def extract_with_fallback(
    image_path: str, prompt: str,
    models: list[str] = None, confidence_threshold: float = 0.7
) -> VLMResult:
    """Extract data with image preprocessing, fallback chain, and cost tracking."""
    if models is None:
        models = ["gpt-4o", "claude-sonnet"]

    image_bytes = preprocess_image(image_path)
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    # Get dimensions for token estimation
    img = Image.open(io.BytesIO(image_bytes))
    img_tokens = estimate_image_tokens_gpt4o(img.width, img.height, "high")

    augmented_prompt = (
        f"{prompt}\n\nAlso include a 'confidence' field (0.0 to 1.0) rating "
        f"how confident you are in the extraction accuracy."
    )

    result = {}
    model_name = models[0]
    output_tokens = 0
    confidence = 0.0
    cost = 0.0

    for model_name in models:
        try:
            # call_openai / call_anthropic wrap the API patterns shown earlier
            if model_name.startswith("gpt"):
                result, output_tokens = call_openai(b64_image, augmented_prompt, model_name)
            elif model_name.startswith("claude"):
                result, output_tokens = call_anthropic(b64_image, augmented_prompt, model_name)
            else:
                continue

            confidence = result.pop("confidence", 0.5)
            pricing = PRICING.get(model_name, PRICING["gpt-4o"])
            cost = (img_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

            if confidence >= confidence_threshold:
                return VLMResult(result, model_name, img_tokens, cost, confidence)

            print(f"  {model_name}: low confidence ({confidence:.0%}), trying next...")
        except Exception as e:
            print(f"  {model_name} failed: {e}, trying next...")

    # All models failed or had low confidence — return best attempt
    return VLMResult(result, model_name, img_tokens, cost, confidence)


def call_openai(b64_image: str, prompt: str, model: str) -> tuple[dict, int]:
    """Call OpenAI vision API. Requires OPENAI_API_KEY."""
    import json
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{b64_image}",
                    "detail": "high"
                }}
            ]
        }],
        max_tokens=2000
    )
    result = json.loads(response.choices[0].message.content)
    output_tokens = response.usage.completion_tokens
    return result, output_tokens


def call_anthropic(b64_image: str, prompt: str, model: str) -> tuple[dict, int]:
    """Call Anthropic vision API. Requires ANTHROPIC_API_KEY."""
    import json
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png",
                    "data": b64_image
                }},
                {"type": "text", "text": prompt}
            ]
        }]
    )
    result = json.loads(response.content[0].text)
    output_tokens = response.usage.output_tokens
    return result, output_tokens


if __name__ == "__main__":
    import tempfile
    import os

    print("=== Production VLM Orchestrator — Self Tests ===\n")

    # Test 1: Token estimation (low detail)
    print("Test 1: Token estimation (low detail)...")
    assert estimate_image_tokens_gpt4o(1920, 1080, "low") == 85
    assert estimate_image_tokens_gpt4o(100, 100, "low") == 85
    assert estimate_image_tokens_gpt4o(4096, 4096, "low") == 85
    print("  All low-detail images: 85 tokens")
    print("  PASS\n")

    # Test 2: Token estimation (high detail) — blog claims 1920x1080 = ~1105 tokens
    print("Test 2: Token estimation (high detail, 1920x1080)...")
    tokens_1080p = estimate_image_tokens_gpt4o(1920, 1080, "high")
    print(f"  1920x1080 at high detail: {tokens_1080p} tokens")
    print(f"  Blog claims: ~1,105 tokens (6 tiles x 170 + 85 base)")
    # Walk through the math:
    # Step 1: scale long side to 2048 → scale = 2048/1920 = 1.066 but min(1.066, 1.0) = 1.0
    # So scaled_w = 1920, scaled_h = 1080
    # Step 2: scale short side to 768 → short_scale = 768/1080 = 0.711
    # scaled_w = int(1920 * 0.711) = 1365, scaled_h = int(1080 * 0.711) = 768
    # Step 3: tiles_w = (1365 + 511) // 512 = 3, tiles_h = (768 + 511) // 512 = 2
    # Total = 85 + (3 * 2 * 170) = 85 + 1020 = 1105
    assert tokens_1080p == 1105, f"Expected 1105, got {tokens_1080p}"
    print("  PASS (matches blog claim of 1,105)\n")

    # Test 3: Token estimation for various resolutions
    print("Test 3: Token estimation for various resolutions...")
    test_cases = [
        (512, 512, "high", 85 + 170),     # 1 tile
        (1024, 1024, "high", 85 + 4*170),  # After short-side scaling to 768: 768x768 → 2x2 = 4 tiles
        (4096, 4096, "high", 85 + 4*170),  # Scale to 2048, then short to 768 → 768x768 → 2x2 = 4 tiles
        (256, 256, "high", 85 + 170),      # Small image, 1 tile
    ]
    for w, h, detail, expected in test_cases:
        actual = estimate_image_tokens_gpt4o(w, h, detail)
        print(f"  {w}x{h} high detail: {actual} tokens (expected {expected})")
        assert actual == expected, f"  FAIL: expected {expected}, got {actual}"
    print("  PASS\n")

    # Test 4: Preprocessing (resize large image)
    print("Test 4: Image preprocessing...")
    # Create a test image
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        test_img = Image.new("RGB", (4096, 3072), color=(128, 64, 32))
        test_img.save(f, format="PNG")
        temp_path = f.name

    preprocessed = preprocess_image(temp_path, max_dimension=2048)
    result_img = Image.open(io.BytesIO(preprocessed))
    print(f"  Original: 4096x3072")
    print(f"  Preprocessed: {result_img.width}x{result_img.height}")
    assert max(result_img.width, result_img.height) <= 2048
    assert result_img.width == 2048  # Long side scaled to max
    assert result_img.height == 1536  # Aspect ratio preserved
    os.unlink(temp_path)
    print("  PASS\n")

    # Test 5: Preprocessing (small image unchanged)
    print("Test 5: Preprocessing small image (no resize)...")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        small_img = Image.new("RGB", (800, 600), color=(255, 128, 0))
        small_img.save(f, format="PNG")
        temp_path = f.name

    preprocessed = preprocess_image(temp_path, max_dimension=2048)
    result_img = Image.open(io.BytesIO(preprocessed))
    print(f"  Original: 800x600")
    print(f"  Preprocessed: {result_img.width}x{result_img.height} (unchanged)")
    assert result_img.width == 800
    assert result_img.height == 600
    os.unlink(temp_path)
    print("  PASS\n")

    # Test 6: Cost calculation
    print("Test 6: Cost calculation...")
    img_tokens = 1105  # 1920x1080 at high detail
    output_tokens = 500
    for model_name, pricing in PRICING.items():
        cost = (img_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
        print(f"  {model_name}: ${cost:.6f} per image (1105 input + 500 output tokens)")
    print("  PASS\n")

    # Test 7: VLMResult dataclass
    print("Test 7: VLMResult dataclass...")
    result = VLMResult(
        data={"text": "hello"},
        model_used="gpt-4o",
        image_tokens=1105,
        cost_usd=0.007763,
        confidence=0.95
    )
    assert result.model_used == "gpt-4o"
    assert result.confidence == 0.95
    print(f"  VLMResult: model={result.model_used}, tokens={result.image_tokens}, "
          f"cost=${result.cost_usd:.6f}, confidence={result.confidence:.0%}")
    print("  PASS\n")

    print("All production orchestrator self-tests passed!")
    print("\nNote: extract_with_fallback() requires API keys to run.")
    print("Tested without API: token estimation, preprocessing, cost math, data structures.")
