import math

class Value:
    """Minimal autograd (same engine from Section 3)."""
    def __init__(self, data, children=()):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._children = set(children)
    def __add__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        out = Value(self.data + o.data, (self, o))
        def _bw():
            self.grad += out.grad; o.grad += out.grad
        out._backward = _bw; return out
    def __mul__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        out = Value(self.data * o.data, (self, o))
        def _bw():
            self.grad += o.data * out.grad; o.grad += self.data * out.grad
        out._backward = _bw; return out
    def __neg__(self): return self * -1
    def __sub__(self, o): return self + (-o)
    def __radd__(self, o): return self + o
    def __rmul__(self, o): return self * o
    def backward(self):
        order, visited = [], set()
        def topo(v):
            if v not in visited:
                visited.add(v)
                for c in v._children: topo(c)
                order.append(v)
        topo(self)
        self.grad = 1.0
        for v in reversed(order): v._backward()

def diff_sin(x):
    out = Value(math.sin(x.data), (x,))
    def _bw(): x.grad += math.cos(x.data) * out.grad
    out._backward = _bw; return out

def diff_cos(x):
    out = Value(math.cos(x.data), (x,))
    def _bw(): x.grad += -math.sin(x.data) * out.grad
    out._backward = _bw; return out

# Differentiable projectile: optimize angle to hit target
target_x = 10.0
v0 = 15.0       # launch speed (fixed)
g = 9.81         # gravity
dt = 0.01        # time step

angle = Value(0.5)  # initial guess: ~28.6 degrees

for step in range(80):
    # Reset gradients
    angle.grad = 0.0

    # Simulate trajectory with Euler integration
    vx = v0 * diff_cos(angle)
    vy = v0 * diff_sin(angle)
    px = Value(0.0)
    py = Value(0.0)

    for t in range(200):
        px = px + vx * dt
        py = py + vy * dt
        vy = vy + Value(-g) * dt
        if py.data < 0 and t > 5:
            break

    # Loss: squared distance from target
    loss = (px - target_x) * (px - target_x)
    loss.backward()

    # Gradient descent on the angle
    angle = Value(angle.data - 0.001 * angle.grad)

    if step % 20 == 0:
        print(f"step {step:3d}  angle={math.degrees(angle.data):6.2f} deg  "
              f"landing={px.data:6.2f}  loss={loss.data:.4f}")
