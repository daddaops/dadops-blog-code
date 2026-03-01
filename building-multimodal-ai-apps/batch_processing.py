"""
Batch image processing with async concurrency and rate limiting.

Blog post: https://dadops.dev/blog/building-multimodal-ai-apps/
Code Block 5.

Requires: openai.
NOTE: Requires OPENAI_API_KEY environment variable.
"""
import asyncio
import base64
import json
from pathlib import Path


CATEGORIES = ["screenshot", "photograph", "diagram", "document", "chart", "other"]


# ── Code Block 5: Batch Image Classification ──

async def process_single_image(
    client, image_path: str, semaphore: asyncio.Semaphore
) -> dict:
    """Process one image with rate limiting via semaphore."""
    async with semaphore:
        image_bytes = Path(image_path).read_bytes()
        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        suffix = Path(image_path).suffix.lstrip(".")
        media_type = f"image/{'jpeg' if suffix == 'jpg' else suffix}"

        for attempt in range(3):
            try:
                response = await client.chat.completions.create(
                    model="gpt-4o-mini",  # Cheaper model for classification
                    response_format={"type": "json_object"},
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": (
                                "Analyze this image and return JSON with:\n"
                                f"- category: one of {CATEGORIES}\n"
                                "- subject: brief description (10 words max)\n"
                                "- has_text: boolean\n"
                                "- text_content: extracted text if any (empty string if none)\n"
                                "- dominant_colors: list of 1-3 color names"
                            )},
                            {"type": "image_url", "image_url": {
                                "url": f"data:{media_type};base64,{b64_image}",
                                "detail": "low"  # Low detail for classification
                            }}
                        ]
                    }],
                    max_tokens=500
                )
                result = json.loads(response.choices[0].message.content)
                result["file"] = image_path
                result["status"] = "success"
                return result
            except Exception as e:
                if attempt == 2:
                    return {"file": image_path, "status": "error", "error": str(e)}
                await asyncio.sleep(2 ** attempt)  # Exponential backoff


async def batch_process_images(image_dir: str, max_concurrent: int = 5) -> list:
    """Process all images in a directory with controlled concurrency."""
    from openai import AsyncOpenAI

    image_paths = [
        str(p) for p in Path(image_dir).iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif"}
    ]
    print(f"Processing {len(image_paths)} images (max {max_concurrent} concurrent)...")

    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [process_single_image(client, p, semaphore) for p in image_paths]

    results = []
    for i, coro in enumerate(asyncio.as_completed(tasks)):
        result = await coro
        results.append(result)
        status = "OK" if result["status"] == "success" else "FAIL"
        print(f"  [{i+1}/{len(tasks)}] {Path(result['file']).name}: {status}")

    successes = sum(1 for r in results if r["status"] == "success")
    print(f"\nDone: {successes}/{len(results)} succeeded")
    return results


if __name__ == "__main__":
    print("=== Batch Image Processing ===\n")
    print("This script requires OPENAI_API_KEY to run.")
    print("Usage: asyncio.run(batch_process_images('path/to/images/', max_concurrent=5))")
    print("No self-tests available without API key.")
