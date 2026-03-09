import numpy as np

# Three major instruction tuning data formats

# 1. Alpaca format -- simple, single-turn
alpaca_examples = [
    {"instruction": "What is the capital of France?",
     "input": "",
     "output": "The capital of France is Paris."},
    {"instruction": "Translate the following to Spanish.",
     "input": "The weather is beautiful today.",
     "output": "El clima esta hermoso hoy."},
    {"instruction": "Write a Python function that checks if a number is prime.",
     "input": "",
     "output": "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"},
    {"instruction": "Summarize the key idea in one sentence.",
     "input": "Photosynthesis converts sunlight into chemical energy...",
     "output": "Plants convert sunlight into usable energy through photosynthesis."},
    {"instruction": "List three benefits of regular exercise.",
     "input": "",
     "output": "1. Improved cardiovascular health\n2. Better mental clarity\n3. Stronger immune system"},
]

# 2. ChatML / OpenAI format -- multi-turn conversations
chatml_example = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "How do I read a CSV file in Python?"},
    {"role": "assistant", "content": "Use pandas: pd.read_csv('file.csv')"},
    {"role": "user", "content": "What if the file has no header?"},
    {"role": "assistant", "content": "Pass header=None: pd.read_csv('file.csv', header=None)"},
]

# Dataset statistics
instructions = [ex["instruction"] for ex in alpaca_examples]
responses = [ex["output"] for ex in alpaca_examples]
avg_inst_len = np.mean([len(s.split()) for s in instructions])
avg_resp_len = np.mean([len(s.split()) for s in responses])

print(f"Dataset size: {len(alpaca_examples)} examples")
print(f"Avg instruction length: {avg_inst_len:.1f} words")
print(f"Avg response length: {avg_resp_len:.1f} words")
print(f"Ratio (response/instruction): {avg_resp_len/avg_inst_len:.1f}x")
# Dataset size: 5 examples
# Avg instruction length: 7.8 words
# Avg response length: 14.8 words
# Ratio (response/instruction): 1.9x
