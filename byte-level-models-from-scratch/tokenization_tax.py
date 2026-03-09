"""Tokenization Tax — comparing byte counts vs subword token counts across languages.

Code Block 1: Shows how subword tokenizers penalize non-Latin scripts.
"""
# Simulated BPE tokenizer: English-biased merge rules
# Real BPE would use learned merges; we simulate the token count pattern
texts = {
    'English': 'Hello, how are you?',
    'Chinese': '你好，你怎么样？',
    'Arabic':  'مرحبا، كيف حالك؟',
    'Thai':    'สวัสดี คุณเป็นอย่างไร?',
    'Code':    'def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)',
}

# Approximate subword token counts (based on GPT-2 tokenizer patterns)
approx_tokens = {'English': 6, 'Chinese': 11, 'Arabic': 16, 'Thai': 24, 'Code': 19}

print(f"{'Language':<10} {'Bytes':>6} {'Tokens':>7} {'Tax':>6}")
print("-" * 32)
for lang, text in texts.items():
    byte_count = len(text.encode('utf-8'))
    token_count = approx_tokens[lang]
    tax = token_count / byte_count
    print(f"{lang:<10} {byte_count:>6} {token_count:>7} {tax:>6.2f}")

# Blog claims:
# English       19       6   0.32
# Chinese       24      11   0.46
# Arabic        30      16   0.53
# Thai          62      24   0.39
# Code          54      19   0.35
