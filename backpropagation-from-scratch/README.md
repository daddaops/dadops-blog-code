# Backpropagation from Scratch

Verified, runnable code from the DadOps blog post:
[Backpropagation from Scratch: How Neural Networks Learn by Going Backwards](https://www.dadops.co/blog/backpropagation-from-scratch/)

## Scripts

- `numerical_gradients.py` — Validates chain rule via numerical vs analytic gradients (finite differences)
- `local_gradients.py` — Tests local gradients for all fundamental operations (add, multiply, ReLU, sigmoid, matmul, softmax+CE)
- `mlp_xor.py` — Full manual backprop on a 3-layer MLP trained on XOR with gradient verification
- `gradient_flow.py` — Empirical study of vanishing/exploding gradients across depths and activation functions

## Run

```bash
pip install -r requirements.txt
python numerical_gradients.py
python local_gradients.py
python mlp_xor.py
python gradient_flow.py
```
