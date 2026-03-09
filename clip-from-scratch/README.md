# CLIP from Scratch

Code extracted from the [CLIP from Scratch](https://dadops.dev/blog/clip-from-scratch/) blog post on DadOps.

## Scripts

| Script | Description |
|--------|-------------|
| `clip_model.py` | CLIP architecture: VisionEncoder (ViT), TextEncoder, and full CLIP model |
| `clip_loss.py` | Symmetric contrastive loss (InfoNCE across modalities) |
| `train_clip.py` | Minimal training loop with synthetic data demo |
| `zero_shot.py` | Zero-shot classification and prompt ensembling |
| `clip_search.py` | Text-to-image and image-to-image search |

## Usage

```bash
pip install -r requirements.txt

# Run each script independently
python clip_model.py      # Smoke test the architecture
python clip_loss.py       # Demo contrastive loss on synthetic embeddings
python train_clip.py      # Train on synthetic data (verifies the loop)
python zero_shot.py       # Zero-shot classification demo
python clip_search.py     # Search demo
```

## Notes

- All scripts use tiny model configs and synthetic data so they run on CPU in seconds.
- The `tokenize()` function is a placeholder — in production, use a BPE tokenizer.
- Model weights are random (untrained), so classification/search results are random. The code demonstrates the correct API and data flow.
