"""Complete micrograd autograd engine.

A scalar-valued autograd engine that supports forward computation
and reverse-mode automatic differentiation. Combines all the code
blocks from the blog post into a single working module.
"""
import math


class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data              # the actual number
        self.grad = 0.0               # derivative of the output w.r.t. this value
        self._backward = lambda: None # function to compute local gradients
        self._prev = set(_children)   # parent Values that produced this one
        self._op = _op                # the operation that created this (for debugging)

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad     # d(a+b)/da = 1, so gradient passes through
            other.grad += out.grad    # d(a+b)/db = 1, same thing
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad   # d(a*b)/da = b
            other.grad += self.data * out.grad   # d(a*b)/db = a
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))  # only constant exponents
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    # Convenience methods — these build on the operators above
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other ** -1

    # Handle cases like  2.0 + Value(3.0)  where Python tries int.__add__ first
    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def backward(self):
        # Build topological order via depth-first search
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v._prev:
                    build_topo(parent)
                topo.append(v)
        build_topo(self)

        # Seed the output gradient
        self.grad = 1.0

        # Walk in reverse: output → inputs
        for v in reversed(topo):
            v._backward()


if __name__ == "__main__":
    # Demo: single neuron forward + backward
    x1 = Value(2.0)
    w1 = Value(-3.0)
    x2 = Value(0.0)
    w2 = Value(1.0)
    b  = Value(6.88)

    # A single neuron: weighted sum + tanh activation
    n = x1*w1 + x2*w2 + b   # graph building silently...
    o = n.tanh()

    print(o)  # Value(data=0.7071, grad=0.0000)

    o.backward()

    print(f"x1.grad = {x1.grad:.4f}")  # -1.5000
    print(f"w1.grad = {w1.grad:.4f}")  #  1.0000
    print(f"x2.grad = {x2.grad:.4f}")  #  0.5000
    print(f"w2.grad = {w2.grad:.4f}")  #  0.0000
    print(f"b.grad  = {b.grad:.4f}")   #  0.5000
