# Softmax & Temperature from Scratch

Runnable Python code extracted from the DadOps blog post:
**[Softmax & Temperature from Scratch: How LLMs Make Choices](https://dadops.io/blog/softmax-temperature-from-scratch/)**

## Scripts

| Script | Description |
|--------|-------------|
| `softmax_basics.py` | Why naive normalization fails; softmax from first principles; shift invariance |
| `overflow_demo.py` | Numerical overflow in naive softmax; the subtract-max stability trick |
| `temperature.py` | Temperature-scaled softmax; sweep showing T vs distribution sharpness |
| `entropy_demo.py` | Shannon entropy as a function of temperature |
| `attention_scaling.py` | The sqrt(d_k) attention scaling factor as an effective temperature |
| `sampling.py` | Top-k filtering, top-p (nucleus) sampling, temperature-nucleus interaction |

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python softmax_basics.py
python overflow_demo.py
python temperature.py
python entropy_demo.py
python attention_scaling.py
python sampling.py
```
