import math

class Dual:
    """A dual number a + b*epsilon for forward-mode autodiff."""
    def __init__(self, val, deriv=0.0):
        self.val = val      # function value
        self.deriv = deriv   # derivative value

    def __add__(self, other):
        o = other if isinstance(other, Dual) else Dual(other)
        return Dual(self.val + o.val, self.deriv + o.deriv)

    def __radd__(self, other):
        return Dual(other).__add__(self)

    def __mul__(self, other):
        o = other if isinstance(other, Dual) else Dual(other)
        return Dual(self.val * o.val, self.val * o.deriv + self.deriv * o.val)

    def __rmul__(self, other):
        return Dual(other).__mul__(self)

    def __pow__(self, n):
        return Dual(self.val ** n, n * self.val ** (n - 1) * self.deriv)

    def __neg__(self):
        return Dual(-self.val, -self.deriv)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return Dual(other) + (-self)

def sin(x):
    if isinstance(x, Dual):
        return Dual(math.sin(x.val), math.cos(x.val) * x.deriv)
    return math.sin(x)

def exp(x):
    if isinstance(x, Dual):
        e = math.exp(x.val)
        return Dual(e, e * x.deriv)
    return math.exp(x)

# Evaluate f(x) = x**3 + sin(x) and its derivative at x = 2.0
x = Dual(2.0, 1.0)  # seed derivative = 1 means df/dx
result = x ** 3 + sin(x)
print(f"f(2.0)  = {result.val:.6f}")   # 8 + sin(2) = 8.909297
print(f"f'(2.0) = {result.deriv:.6f}") # 3*4 + cos(2) = 11.583853
