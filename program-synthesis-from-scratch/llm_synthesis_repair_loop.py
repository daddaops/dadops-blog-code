"""LLM-as-Synthesizer self-repair loop.

CEGIS-style loop using a mock LLM as the program generator,
with automatic test-driven self-repair.
"""
def mock_llm(prompt, attempt=0):
    """Simulates LLM code generation with intentional first-attempt bugs."""
    if "absolute difference" in prompt:
        if attempt == 0:
            return "def solve(a, b): return a - b"  # Bug: no abs()
        return "def solve(a, b): return abs(a - b)"  # Fixed
    if "fibonacci" in prompt:
        if attempt == 0:
            return "def solve(n):\n  if n <= 1: return n\n  return solve(n-1) + solve(n-2)"
        return "def solve(n):\n  a, b = 0, 1\n  for _ in range(n): a, b = b, a+b\n  return a"
    return "def solve(x): return x"

def run_tests(code, tests):
    """Execute code against test cases, return (pass, error_msg)."""
    namespace = {}
    try:
        exec(code, namespace)
    except Exception as e:
        return False, f"Syntax error: {e}"
    fn = namespace.get('solve')
    for inputs, expected in tests:
        try:
            result = fn(*inputs) if isinstance(inputs, tuple) else fn(inputs)
        except Exception as e:
            return False, f"Runtime error on input {inputs}: {e}"
        if result != expected:
            return False, f"Wrong answer: solve({inputs}) = {result}, expected {expected}"
    return True, "All tests passed"

def synthesis_loop(spec, tests, max_attempts=3):
    """CEGIS-style loop with LLM as the synthesizer."""
    error_context = ""
    for attempt in range(max_attempts):
        prompt = f"Write a function `solve` that: {spec}"
        if error_context:
            prompt += f"\nPrevious attempt failed: {error_context}\nFix the bug."
        code = mock_llm(prompt, attempt)
        print(f"Attempt {attempt+1}: {code.split(chr(10))[0]}...")
        passed, msg = run_tests(code, tests)
        print(f"  Result: {msg}")
        if passed:
            return code, attempt + 1
        error_context = msg
    return None, max_attempts

# Synthesize "absolute difference" with self-repair
code, attempts = synthesis_loop(
    "computes the absolute difference of two numbers",
    [((5, 3), 2), ((3, 7), 4), ((0, 0), 0), ((-2, 3), 5)]
)
print(f"\nSynthesized in {attempts} attempt(s)")
