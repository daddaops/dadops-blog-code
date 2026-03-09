# FFN from Scratch

Verified, runnable code from the [FFN from Scratch](https://dadops.dev/blog/ffn-from-scratch/) blog post.

## Scripts

- **classic_ffn.py** — Original transformer FFN with ReLU
- **activations.py** — ReLU, GELU, SiLU comparison with gradients
- **gated_ffn.py** — GLU, ReGLU, GEGLU, SwiGLU variants
- **param_budget.py** — Parameter budget analysis (h = 8d/3)
- **swiglu_class.py** — Complete SwiGLU FFN class (LLaMA-style)
- **ffn_memory.py** — FFN as key-value memory interpretation
- **transformer_block.py** — Complete transformer block with attention + SwiGLU

## Run

```bash
pip install -r requirements.txt
python classic_ffn.py
python activations.py
python gated_ffn.py
python param_budget.py
python swiglu_class.py
python ffn_memory.py
python transformer_block.py
```
