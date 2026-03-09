"""CEGIS: Counter-Example Guided Inductive Synthesis.

Starts with minimal examples, synthesizes a candidate,
verifies against random inputs, adds counterexamples, repeats.
"""
import itertools
import random

# AST node types (same as enumerative_synthesis.py)
class Const:
    def __init__(self, v): self.v = v
    def eval(self, env): return self.v
    def __repr__(self): return str(self.v)

class Var:
    def __init__(self, name): self.name = name
    def eval(self, env): return env[self.name]
    def __repr__(self): return self.name

class BinOp:
    def __init__(self, op, l, r): self.op, self.l, self.r = op, l, r
    def eval(self, env):
        a, b = self.l.eval(env), self.r.eval(env)
        if self.op == '+': return a + b
        if self.op == '*': return a * b
        if self.op == '-': return a - b
    def __repr__(self): return f"({self.l} {self.op} {self.r})"

def enumerate_programs(max_depth, var_names, ops=['+', '*', '-']):
    test_envs = [{v: i+1 for i, v in enumerate(var_names)} for _ in range(1)]
    test_envs += [{v: i*3+2 for i, v in enumerate(var_names)}]
    seen_outputs = set()
    def signature(prog):
        try: return tuple(prog.eval(e) for e in test_envs)
        except: return None
    def add_if_new(prog, bank):
        sig = signature(prog)
        if sig is not None and sig not in seen_outputs:
            seen_outputs.add(sig)
            bank.append(prog)
    bank = []
    for c in [0, 1, 2]: add_if_new(Const(c), bank)
    for v in var_names: add_if_new(Var(v), bank)
    yield from bank
    prev_banks = [bank]
    for depth in range(2, max_depth + 1):
        new_bank = []
        all_prev = [p for b in prev_banks for p in b]
        for op in ops:
            for l, r in itertools.product(all_prev, repeat=2):
                add_if_new(BinOp(op, l, r), new_bank)
        prev_banks.append(new_bank)
        yield from new_bank

def synthesize(examples, var_names=['x'], max_depth=4, ops=['+', '*', '-']):
    count = 0
    for prog in enumerate_programs(max_depth, var_names, ops):
        count += 1
        if all(prog.eval({var_names[0]: x}) == y for x, y in examples):
            return prog, count
    return None, count

def cegis_synthesize(ground_truth, var_names=['x'], max_depth=4,
                     ops=['+', '*', '-'], seed=42):
    """CEGIS loop: synthesize, verify, add counterexamples, repeat."""
    rng = random.Random(seed)
    # Start with a small seed set of examples (just 2 — ambiguous!)
    examples = [(0, ground_truth(0)), (1, ground_truth(1))]
    round_num = 0
    total_candidates = 0

    while True:
        round_num += 1
        # SYNTHESIZE: find a program consistent with current examples
        prog, count = synthesize(examples, var_names, max_depth, ops)
        total_candidates += count
        if prog is None:
            return None, round_num, total_candidates

        # VERIFY: test against many random inputs
        counterexample = None
        for _ in range(200):
            x = rng.randint(-50, 50)
            expected = ground_truth(x)
            try:
                actual = prog.eval({'x': x})
            except:
                counterexample = x
                break
            if actual != expected:
                counterexample = x
                break

        if counterexample is None:
            # No counterexample found — program is likely correct
            return prog, round_num, total_candidates

        # REFINE: add the counterexample and loop
        cx = counterexample
        try:
            got = prog.eval({'x': cx})
        except:
            got = "error"
        examples.append((cx, ground_truth(cx)))
        print(f"  Round {round_num}: found {prog}, "
              f"counterexample x={cx} (expected {ground_truth(cx)}, "
              f"got {got})")

# Synthesize f(x) = x^2 + 1 via CEGIS
target = lambda x: x * x + 1
prog, rounds, candidates = cegis_synthesize(target)
print(f"\nCEGIS result: f(x) = {prog}")
print(f"Converged in {rounds} rounds, {candidates} total candidates")
