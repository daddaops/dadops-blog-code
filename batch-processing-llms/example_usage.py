"""
Example usage of BatchProcessor + OpenAI Batch API workflow.

Blog post: https://dadops.dev/blog/batch-processing-llms/
Code Blocks 10 & 11: BatchProcessor usage and OpenAI Batch API.

REQUIRES: OpenAI API key (set OPENAI_API_KEY environment variable)
"""
import asyncio
import json
import os
import time

from batch_processor import BatchProcessor


def run_batch_processor_example():
    """Code Block 10: BatchProcessor usage with product classification."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("SKIP: Set OPENAI_API_KEY to run this example")
        return

    processor = BatchProcessor(
        model="gpt-4o-mini",
        api_key=api_key,
        max_concurrent=15,
        budget=5.00,                          # halt at $5
        checkpoint_path="classify_checkpoint.jsonl"
    )

    products = [
        {"id": "SKU-001", "name": "Organic Baby Spinach 5oz"},
        {"id": "SKU-002", "name": "Clorox Disinfecting Wipes 75ct"},
        # In a real run, this would be ~5,000 items
    ]

    def make_prompt(item):
        return [
            {"role": "system", "content": "Classify the product into exactly "
             "one category: produce, cleaning, dairy, meat, snacks, beverages, "
             "frozen, bakery, other. Respond with just the category name."},
            {"role": "user", "content": item["name"]}
        ]

    results = asyncio.run(
        processor.run(products, make_prompt, id_fn=lambda x: x["id"])
    )

    print(f"\nDone! {processor.calls} API calls, ${processor.cost:.4f} total")
    print(f"Cache hit rate: {processor.cache_hits}/"
          f"{processor.calls + processor.cache_hits}")


def run_batch_api_example():
    """Code Block 11: OpenAI Batch API workflow.

    This submits a batch job to OpenAI's Batch API (50% discount).
    Results arrive asynchronously (typically within 1 hour).
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("SKIP: Set OPENAI_API_KEY to run Batch API example")
        return

    from openai import OpenAI
    client = OpenAI()

    products = [
        {"name": "Organic Baby Spinach 5oz"},
        {"name": "Clorox Disinfecting Wipes 75ct"},
    ]

    # 1. Build the JSONL request file
    requests = []
    for i, product in enumerate(products):
        requests.append({
            "custom_id": f"item-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "Classify into one category: "
                     "produce, cleaning, dairy, meat, snacks, beverages, "
                     "frozen, bakery, other."},
                    {"role": "user", "content": product["name"]}
                ]
            }
        })

    with open("batch_input.jsonl", "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    # 2. Upload and submit
    batch_file = client.files.create(
        file=open("batch_input.jsonl", "rb"), purpose="batch"
    )
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    print(f"Batch submitted: {batch_job.id}")

    # 3. Poll for completion
    while True:
        status = client.batches.retrieve(batch_job.id)
        print(f"Status: {status.status} "
              f"({status.request_counts.completed}"
              f"/{status.request_counts.total})")
        if status.status in ("completed", "failed", "expired"):
            break
        time.sleep(30)

    # 4. Download results
    if status.status == "completed":
        content = client.files.content(status.output_file_id)
        with open("batch_output.jsonl", "wb") as f:
            f.write(content.content)
        print("Results saved to batch_output.jsonl")


if __name__ == "__main__":
    print("=== BatchProcessor Example ===")
    run_batch_processor_example()
    # Uncomment to also run the Batch API example:
    # print("\n=== OpenAI Batch API Example ===")
    # run_batch_api_example()
