"""FlashFill-style string synthesis.

Template-based program synthesis for text transformations,
similar to Excel's FlashFill feature.
"""
# String DSL: template-based candidates for text transformation
TEMPLATES = [
    ("first_word",       lambda s: s.split(" ")[0]),
    ("last_word",        lambda s: s.split(" ")[-1]),
    ("initial_dot_last", lambda s: s[0] + ". " + s.split(" ")[-1]),
    ("upper_first",      lambda s: s.split(" ")[0].upper()),
    ("last_comma_first", lambda s: s.split(" ")[-1] + ", " + s.split(" ")[0]),
    ("first_initial",    lambda s: s[0] + "."),
    ("initials",         lambda s: ". ".join(w[0] for w in s.split(" ")) + "."),
    ("swap_words",       lambda s: " ".join(s.split(" ")[::-1])),
    ("lower_all",        lambda s: s.lower()),
    ("title_case",       lambda s: s.title()),
]

def synth_string(examples):
    """Find the first template consistent with all I/O examples."""
    for name, fn in TEMPLATES:
        if all(fn(inp) == out for inp, out in examples):
            return name, fn
    return None, None

# "John Smith" -> "J. Smith", "Jane Doe" -> "J. Doe"
examples = [("John Smith", "J. Smith"), ("Jane Doe", "J. Doe")]
name, fn = synth_string(examples)
print(f"Synthesized: {name}")
print(f"Test: 'Alice Cooper' -> '{fn('Alice Cooper')}'")
# Synthesized: initial_dot_last
# Test: 'Alice Cooper' -> 'A. Cooper'
