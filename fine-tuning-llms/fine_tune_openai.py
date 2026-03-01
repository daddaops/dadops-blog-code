"""
Fine-tune GPT-4o-mini via OpenAI API.

Blog post: https://dadops.dev/blog/fine-tuning-llms/
Code Block 4.

Uploads training data, creates a fine-tuning job, polls for completion,
and tests the resulting model.

REQUIRES: OpenAI API key (set OPENAI_API_KEY environment variable)

Blog claims:
  - Training cost: ~$1.20 for 200 examples (~500 tokens each, 4 epochs)
  - Training time: 10-20 minutes
"""
from openai import OpenAI
import time

client = OpenAI()


if __name__ == "__main__":
    # Step 1: Upload training file
    training_file = client.files.create(
        file=open("training_data.jsonl", "rb"),
        purpose="fine-tune"
    )
    print(f"File uploaded: {training_file.id}")

    # Step 2: Create fine-tuning job
    job = client.fine_tuning.jobs.create(
        training_file=training_file.id,
        model="gpt-4o-mini-2024-07-18",
        hyperparameters={
            "n_epochs": 3,                 # 2-4 for most tasks
            # "learning_rate_multiplier": 1.0  # usually auto is fine
        },
        suffix="ticket-classifier"         # your model gets a readable name
    )
    print(f"Job created: {job.id}")

    # Step 3: Wait for completion
    while True:
        status = client.fine_tuning.jobs.retrieve(job.id)
        print(f"Status: {status.status}")
        if status.status in ("succeeded", "failed"):
            break
        time.sleep(60)

    # Step 4: Use your fine-tuned model
    fine_tuned_model = status.fine_tuned_model
    print(f"Model ready: {fine_tuned_model}")

    response = client.chat.completions.create(
        model=fine_tuned_model,  # Use just like any other model
        messages=[
            {"role": "system", "content": "Extract structured data from support tickets. Return valid JSON with fields: category, priority, platform, feature, sentiment."},
            {"role": "user", "content": "Subject: Payment failed\nBody: Tried to upgrade to Pro plan but payment keeps getting declined. Using Visa. Tried 3 times."}
        ]
    )
    print(response.choices[0].message.content)
    # {"category": "billing", "priority": "high", "platform": "unknown", "feature": "payments", "sentiment": "frustrated"}
