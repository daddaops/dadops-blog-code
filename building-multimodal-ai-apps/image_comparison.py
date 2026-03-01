"""
Image comparison: detect visual differences between two images using a VLM.

Blog post: https://dadops.dev/blog/building-multimodal-ai-apps/
Code Block 6.

Requires: openai.
NOTE: Requires OPENAI_API_KEY environment variable.
"""
import base64
import json
from pathlib import Path


# ── Code Block 6: Image Comparison ──

def compare_images(image_path_1: str, image_path_2: str) -> dict:
    """Compare two images and identify differences using a VLM."""
    from openai import OpenAI

    images = []
    for path in [image_path_1, image_path_2]:
        img_bytes = Path(path).read_bytes()
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        suffix = Path(path).suffix.lstrip(".")
        media_type = f"image/{'jpeg' if suffix == 'jpg' else suffix}"
        images.append({"type": "image_url", "image_url": {
            "url": f"data:{media_type};base64,{b64}", "detail": "high"
        }})

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Image 1 (before):"},
                images[0],
                {"type": "text", "text": "Image 2 (after):"},
                images[1],
                {"type": "text", "text": (
                    "Compare these two images carefully.\n"
                    "1. First, describe Image 1 in detail.\n"
                    "2. Then, describe Image 2 in detail.\n"
                    "3. List every difference you can find.\n\n"
                    "Return JSON with: description_1, description_2, "
                    "differences (list of strings), severity "
                    "(none/minor/major), summary (one sentence)."
                )}
            ]
        }],
        max_tokens=1500
    )
    return json.loads(response.choices[0].message.content)


if __name__ == "__main__":
    print("=== Image Comparison ===\n")
    print("This script requires OPENAI_API_KEY to run.")
    print("Usage: compare_images('before.png', 'after.png')")
    print("No self-tests available without API key.")
