# Flash Attention from Scratch

Verified, runnable code from the [Flash Attention from Scratch](https://dadops.dev/blog/flash-attention-from-scratch/) blog post.

## Scripts

- **naive_attention.py** — Standard attention with HBM access counting
- **online_softmax.py** — Online softmax (standard vs streaming comparison)
- **flash_attention.py** — Flash Attention core algorithm with exactness proof
- **hbm_comparison.py** — HBM access counting (naive vs flash, bandwidth savings)

## Run

```bash
pip install -r requirements.txt
python naive_attention.py
python online_softmax.py
python flash_attention.py
python hbm_comparison.py
```
