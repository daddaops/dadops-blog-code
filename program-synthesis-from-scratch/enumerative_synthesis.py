"""Enumerative program synthesis with observational equivalence pruning.

Bottom-up enumeration over an integer DSL (constants, variables, +, *, -)
to find the simplest program matching I/O examples.
"""
import itertools

# AST node types for our integer DSL
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
    """Bottom-up enumeration with observational equivalence pruning."""
    test_envs = [{v: i+1 for i, v in enumerate(var_names)} for _ in range(1)]
    test_envs += [{v: i*3+2 for i, v in enumerate(var_names)}]  # two probe envs
    seen_outputs = set()

    def signature(prog):
        try: return tuple(prog.eval(e) for e in test_envs)
        except: return None

    def add_if_new(prog, bank):
        sig = signature(prog)
        if sig is not None and sig not in seen_outputs:
            seen_outputs.add(sig)
            bank.append(prog)

    # Depth 1: constants and variables
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
    """Find the simplest program consistent with all I/O examples."""
    count = 0
    for prog in enumerate_programs(max_depth, var_names, ops):
        count += 1
        if all(prog.eval({var_names[0]: x}) == y for x, y in examples):
            return prog, count
    return None, count

# Example: synthesize f(x) = 2x + 1 from three examples
prog, n = synthesize([(1, 3), (3, 7), (5, 11)])
print(f"Found: f(x) = {prog}  (explored {n} candidates)")
# Found: f(x) = (1 + (x + x))  (explored 20 candidates)
