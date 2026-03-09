# Contrastive Learning from Scratch — Code Extracts

Code blocks extracted from the [DadOps blog post](../../blog/contrastive-learning-from-scratch/).

## Scripts

| # | File | Description |
|---|------|-------------|
| 1 | `01_simclr_encoder.py` | SimCLR encoder + projection head architecture |
| 2 | `02_info_nce_loss.py` | NT-Xent (InfoNCE) contrastive loss function |
| 3 | `03_augmented_batch.py` | Data augmentation to create positive pairs |
| 4 | `04_train_simclr.py` | Full SimCLR training loop with numerical gradients |
| 5 | `05_evaluate_representations.py` | Evaluate learned representations via cluster similarity |
| 6 | `06_clip_loss.py` | CLIP-style symmetric image-text contrastive loss |
| 7 | `07_dino_trainer.py` | DINO self-distillation framework (EMA teacher, centering) |

## Requirements

```
pip install -r requirements.txt
```

Only dependency: `numpy`

## Running

Each script is self-contained and runnable independently:

```bash
python3 01_simclr_encoder.py
python3 02_info_nce_loss.py
# ... etc.
```

Note: Scripts 04 and 05 use numerical gradients and may take a minute to run.
