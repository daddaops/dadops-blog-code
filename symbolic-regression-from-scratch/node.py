"""Shared Node class for expression trees."""
import numpy as np


class Node:
    """A single node in an expression tree."""
    def __init__(self, value, children=None):
        self.value = value
        self.children = children or []

    def evaluate(self, x):
        """Evaluate the tree for input value(s) x (numpy array)."""
        if not self.children:
            if self.value == 'x':
                return x
            return np.full_like(x, float(self.value), dtype=float)

        args = [c.evaluate(x) for c in self.children]
        if self.value == '+':   return args[0] + args[1]
        if self.value == '-':   return args[0] - args[1]
        if self.value == '*':   return args[0] * args[1]
        if self.value == '/':
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
