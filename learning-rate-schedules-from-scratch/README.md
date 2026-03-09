# Learning Rate Schedules from Scratch

Verified, runnable code from the [Learning Rate Schedules from Scratch](https://dadops.dev/blog/learning-rate-schedules-from-scratch/) blog post.

## Scripts

- **training_basics.py** — Shared module: MLP, spiral data, constant LR demo
- **classic_schedules.py** — Step decay, exponential decay, inverse sqrt decay
- **cosine_annealing.py** — Cosine annealing and cosine with warm restarts (SGDR)
- **warmup_cosine.py** — Linear warmup + cosine decay (the LLM recipe)
- **lr_range_test.py** — Leslie Smith's LR range test
- **grand_comparison.py** — All 6 schedules head-to-head

## Run

```bash
pip install -r requirements.txt
python training_basics.py
python classic_schedules.py
python cosine_annealing.py
python warmup_cosine.py
python lr_range_test.py
python grand_comparison.py
```
