# DPO from Scratch

Verified, runnable code from the [DPO from Scratch](https://dadops.dev/blog/dpo-from-scratch/) blog post.

## Scripts

- **reward_model.py** — Bradley-Terry reward model training (the RLHF component DPO eliminates)
- **dpo_training.py** — Core DPO implementation: ToyLM, DPO loss, training loop, preference evaluation
- **dpo_vs_rlhf.py** — Side-by-side comparison of DPO vs simplified PPO-style RLHF

## Run

```bash
pip install -r requirements.txt
python reward_model.py
python dpo_training.py
python dpo_vs_rlhf.py
```
