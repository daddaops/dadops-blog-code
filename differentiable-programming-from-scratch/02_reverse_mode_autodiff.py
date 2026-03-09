class Value:
    """A node in a computational graph for reverse-mode autodiff."""
    def __init__(self, data, children=(), op=''):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._children = set(children)
        self._op = op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, n):
        out = Value(self.data ** n, (self,), f'**{n}')
        def _backward():
            self.grad += n * self.data ** (n - 1) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        return self + (-other)

    def sin(self):
        import math
        out = Value(math.sin(self.data), (self,), 'sin')
        def _backward():
            self.grad += math.cos(self.data) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        import math
        e = math.exp(self.data)
        out = Value(e, (self,), 'exp')
        def _backward():
            self.grad += e * out.grad
        out._backward = _backward
        return out

    def backward(self):
        order = []
        visited = set()
        def topo(v):
            if v not in visited:
                visited.add(v)
                for c in v._children:
                    topo(c)
                order.append(v)
        topo(self)
        self.grad = 1.0
        for v in reversed(order):
            v._backward()

# Differentiate a non-neural-network function:
# f(x, y) = sin(x * y) + exp(x^2)
x = Value(1.0)
y = Value(2.0)
f = (x * y).sin() + (x ** 2).exp()

f.backward()
print(f"f(1,2)   = {f.data:.6f}")   # sin(2) + exp(1) = 3.627579
print(f"df/dx    = {x.grad:.6f}")   # y*cos(xy) + 2x*exp(x^2) = 4.604270
print(f"df/dy    = {y.grad:.6f}")   # x*cos(xy) = cos(2) = -0.416147
