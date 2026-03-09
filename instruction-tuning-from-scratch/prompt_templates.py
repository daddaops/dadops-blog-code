def apply_template(instruction, response, template_name):
    """Convert an instruction-response pair into a formatted training sequence.
    Returns (formatted_text, response_start_position)."""

    if template_name == "chatml":
        # OpenAI / ChatML format
        prefix = (
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            f"{instruction}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        full = prefix + response + "<|im_end|>"

    elif template_name == "llama2":
        # Meta Llama-2 chat format
        prefix = (
            f"[INST] <<SYS>>\nYou are a helpful assistant.\n"
            f"<</SYS>>\n\n{instruction} [/INST] "
        )
        full = prefix + response

    elif template_name == "alpaca":
        # Stanford Alpaca format
        prefix = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n"
        )
        full = prefix + response

    return full, len(prefix)

# Compare templates for the same instruction
instruction = "Explain recursion in one sentence."
response = "Recursion is when a function calls itself to solve smaller subproblems."

for tmpl in ["chatml", "llama2", "alpaca"]:
    text, split = apply_template(instruction, response, tmpl)
    prefix_part = text[:split]
    response_part = text[split:]
    print(f"--- {tmpl.upper()} (response starts at char {split}) ---")
    print(f"INSTRUCTION: {prefix_part[:80]}...")
    print(f"RESPONSE:    {response_part}")
    print(f"Total chars: {len(text)}\n")
# --- CHATML (response starts at char 96) ---
# INSTRUCTION: <|im_start|>system
# You are a helpful assistant.<|im_end|>
# <|im_start|>user
# Explain recu...
# RESPONSE:    Recursion is when a function calls itself to solve smaller subproblems.<|im_end|>
# Total chars: 178
