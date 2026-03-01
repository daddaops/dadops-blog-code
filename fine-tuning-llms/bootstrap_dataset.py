"""
Bootstrap training dataset using GPT-4o as teacher model.

Blog post: https://dadops.dev/blog/fine-tuning-llms/
Code Block 3.

Takes 20 hand-written seed examples, generates 10 variations each,
and labels them using GPT-4o for a total of 200 training examples.

REQUIRES: OpenAI API key (set OPENAI_API_KEY environment variable)
"""
from openai import OpenAI
import json

client = OpenAI()

# Your 20 hand-written seed examples (abbreviated)
seed_tickets = [
    "Subject: App crashes on upload\nBody: File upload crashes on iOS...",
    "Subject: Billing error\nBody: Charged twice for subscription...",
    "Subject: Slow performance\nBody: Dashboard takes 30s to load...",
    # ... 17 more diverse examples
]


def generate_variations(seed, n=10):
    """Generate n realistic ticket variations from a seed example."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "system",
            "content": "Generate realistic customer support tickets. "
                       "Vary the writing style, detail level, and tone. "
                       "Return one ticket per line as: Subject: ...\nBody: ..."
        }, {
            "role": "user",
            "content": f"Generate {n} variations of this ticket:\n{seed}"
        }],
        temperature=0.9
    )
    return response.choices[0].message.content.strip().split("\n\n")


# Generate, then label each variation with GPT-4o
def label_ticket(ticket_text):
    """Use GPT-4o to label a ticket (the 'teacher' model)."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Extract structured data from support tickets. Return ONLY valid JSON with fields: category, priority, platform, feature, sentiment."},
            {"role": "user", "content": ticket_text}
        ],
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    # Build the dataset
    training_data = []
    for seed in seed_tickets:
        variations = generate_variations(seed, n=10)
        for ticket in variations:
            label = label_ticket(ticket)
            training_data.append({
                "messages": [
                    {"role": "system", "content": "Extract structured data from support tickets. Return valid JSON with fields: category, priority, platform, feature, sentiment."},
                    {"role": "user", "content": ticket},
                    {"role": "assistant", "content": label}
                ]
            })

    # Save as JSONL — always review a random sample before training!
    with open("training_data.jsonl", "w") as f:
        for example in training_data:
            f.write(json.dumps(example) + "\n")
    print(f"Generated {len(training_data)} training examples")
