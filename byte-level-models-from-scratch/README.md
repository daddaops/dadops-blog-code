# Byte-Level Models from Scratch

Verified, runnable code from the [Byte-Level Models from Scratch](https://dadops.co/blog/byte-level-models-from-scratch/) blog post.

## Scripts

| Script | Description |
|--------|-------------|
| `tokenization_tax.py` | Byte vs token counts across languages |
| `utf8_encoding.py` | UTF-8 encoding from first principles |
| `byt5_encoder.py` | ByT5-style byte encoder with downsampling |
| `megabyte_costs.py` | MegaByte attention cost analysis |
| `architecture_comparison.py` | Attention vs MegaByte vs SSM scaling |
| `robustness.py` | Typo robustness: bytes vs subword tokens |

## Quick Start

```bash
pip install -r requirements.txt
python tokenization_tax.py
python utf8_encoding.py
python byt5_encoder.py
python megabyte_costs.py
python architecture_comparison.py
python robustness.py
```
