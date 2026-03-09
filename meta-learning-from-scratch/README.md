# Meta-Learning from Scratch

Verified, runnable code from the [DadOps blog post](https://dadops.co/blog/meta-learning-from-scratch/).

## Scripts

| Script | Description |
|--------|-------------|
| `few_shot_failure.py` | Shows why standard training fails on 5-way 5-shot |
| `prototypical_network.py` | Prototypical Network with episodic training (full backprop) |
| `maml_sinusoid.py` | MAML for few-shot sinusoid regression |
| `fomaml_reptile.py` | FOMAML and Reptile first-order approximations |
| `spectrum.py` | Few-shot adaptation spectrum comparison table |

## Usage

```bash
pip install -r requirements.txt
python few_shot_failure.py
python prototypical_network.py
python maml_sinusoid.py
python fomaml_reptile.py
python spectrum.py
```
