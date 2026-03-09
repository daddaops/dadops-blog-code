# LoRA from Scratch

Verified, runnable code from the [LoRA from Scratch](https://dadops.dev/blog/lora-from-scratch/) blog post.

## Scripts

- **svd_energy.py** — SVD analysis showing weight updates are low-rank
- **lora_layer.py** — LoRALayer class with parameter count comparison
- **lora_training.py** — Full demo: base MLP training + LoRA adaptation on rotated spirals
- **merging.py** — Weight merging for deployment + storage economics

## Run

```bash
pip install -r requirements.txt
python svd_energy.py
python lora_layer.py
python lora_training.py
python merging.py
```
