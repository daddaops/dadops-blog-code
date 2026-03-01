"""
Two-step chart analysis: describe then answer, using Anthropic Claude.

Blog post: https://dadops.dev/blog/building-multimodal-ai-apps/
Code Block 3.

Requires: anthropic.
NOTE: Requires ANTHROPIC_API_KEY environment variable.
"""
import base64
from pathlib import Path


# ── Code Block 3: Two-Step Chart Analysis ──

def analyze_chart(image_path: str, question: str) -> dict:
    """Analyze a chart image with multi-turn visual reasoning."""
    import anthropic

    image_bytes = Path(image_path).read_bytes()
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    suffix = Path(image_path).suffix.lstrip(".")
    media_type = f"image/{'jpeg' if suffix == 'jpg' else suffix}"

    client = anthropic.Anthropic()

    # Step 1: Describe the chart structure (grounding)
    description_response = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": media_type,
                    "data": b64_image
                }},
                {"type": "text", "text": (
                    "Describe this chart in detail: chart type, axes labels, "
                    "data series, approximate values for each data point, "
                    "and any notable trends."
                )}
            ]
        }]
    )
    chart_description = description_response.content[0].text

    # Step 2: Answer the question grounded in the description
    answer_response = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=500,
        messages=[
            {"role": "user", "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": media_type,
                    "data": b64_image
                }},
                {"type": "text", "text": (
                    "Describe this chart in detail."
                )}
            ]},
            {"role": "assistant", "content": chart_description},
            {"role": "user", "content": (
                f"Based on your description of the chart, answer this question: "
                f"{question}\n\nProvide the answer and your confidence (low/medium/high)."
            )}
        ]
    )
    return {
        "chart_description": chart_description,
        "answer": answer_response.content[0].text,
        "tokens_used": (description_response.usage.input_tokens +
                        description_response.usage.output_tokens +
                        answer_response.usage.input_tokens +
                        answer_response.usage.output_tokens)
    }


if __name__ == "__main__":
    print("=== Chart Analysis ===\n")
    print("This script requires ANTHROPIC_API_KEY to run.")
    print("All functions are API-dependent (Claude Sonnet).")
    print("No self-tests available without API key.")
