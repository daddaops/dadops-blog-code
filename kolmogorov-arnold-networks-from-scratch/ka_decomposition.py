import numpy as np

# Target: f(x, y) = sin(x) * exp(y)
# KA-style decomposition: rewrite as univariate functions + addition
# sin(x) * exp(y) = exp(log(sin(x)) + y)  [for sin(x) > 0]
# This is: outer(inner_1(x) + inner_2(y))
# where inner_1(x) = log(sin(x)), inner_2(y) = y, outer(t) = exp(t)

x = np.linspace(0.1, 3.0, 50)
y = np.linspace(-1.0, 1.0, 50)
X, Y = np.meshgrid(x, y)

# Direct computation
f_direct = np.sin(X) * np.exp(Y)

# KA-style decomposition: composition of univariate functions
inner_1 = np.log(np.sin(X))   # univariate in x
inner_2 = Y                    # univariate in y (identity)
outer = np.exp(inner_1 + inner_2)  # univariate outer function

max_error = np.max(np.abs(f_direct - outer))
print(f"Max error between direct and KA decomposition: {max_error:.2e}")
# Output: Max error between direct and KA decomposition: 4.44e-16

# MLP approach: approximate with matrix multiply + ReLU
# Would need hundreds of parameters to reach this precision
# KA decomposition uses exactly 3 univariate functions
