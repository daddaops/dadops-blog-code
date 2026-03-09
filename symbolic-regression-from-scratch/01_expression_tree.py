"""Expression tree representation for symbolic regression."""
import numpy as np


class Node:
    """A single node in an expression tree."""
    def __init__(self, value, children=None):
        self.value = value          # operator string or float/str terminal
        self.children = children or []

    def evaluate(self, x):
        """Evaluate the tree for input value(s) x (numpy array)."""
        if not self.children:
            # Terminal: variable or constant
            if self.value == 'x':
                return x
            return np.full_like(x, float(self.value), dtype=float)

        args = [c.evaluate(x) for c in self.children]
        if self.value == '+':   return args[0] + args[1]
        if self.value == '-':   return args[0] - args[1]
        if self.value == '*':   return args[0] * args[1]
        if self.value == '/':   # protected division
            safe = np.where(np.abs(args[1]) < 1e-10, 1.0, args[1])
            return args[0] / safe
        if self.value == 'sin': return np.sin(args[0])
        return np.zeros_like(x)

    def size(self):
        return 1 + sum(c.size() for c in self.children)

    def depth(self):
        if not self.children:
            return 0
        return 1 + max(c.depth() for c in self.children)

    def __str__(self):
        if not self.children:
            if isinstance(self.value, float):
                return f"{self.value:.2f}"
            return str(self.value)
        if self.value in ('+', '-', '*', '/'):
            return f"({self.children[0]} {self.value} {self.children[1]})"
        return f"{self.value}({self.children[0]})"


if __name__ == "__main__":
    # Demo: build and evaluate a simple expression tree: (x + 2.0)
    tree = Node('+', [Node('x'), Node(2.0)])
    x = np.linspace(-3, 3, 7)
    print("Expression:", tree)
    print("x:", x)
    print("f(x):", tree.evaluate(x))
    print("Size:", tree.size())
    print("Depth:", tree.depth())

    # More complex tree: sin(x) * 3.0
    tree2 = Node('*', [Node('sin', [Node('x')]), Node(3.0)])
    print("\nExpression:", tree2)
    print("f(x):", tree2.evaluate(x))
    print("Size:", tree2.size())
    print("Depth:", tree2.depth())
