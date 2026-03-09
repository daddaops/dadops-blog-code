# State Space Models from Scratch — Code Extracts

Standalone Python scripts extracted from the [DadOps blog post](https://dadops.dev/blog/state-space-models-from-scratch/).

## Scripts

| # | Script | Description |
|---|--------|-------------|
| 1 | `01_continuous_ssm.py` | Continuous-time SSM simulation with Euler's method — 4D oscillatory dynamics driven by a pulse input |
| 2 | `02_discretization_zoh.py` | Zero-Order Hold discretization — converts continuous SSM to discrete recurrence and verifies accuracy |
| 3 | `03_convolution_kernel.py` | Dual computation trick — computes SSM output via FFT convolution and sequential recurrence, proving they match |
| 4 | `04_hippo_memory.py` | HiPPO initialization vs random — demonstrates long-range memory retention with Legendre polynomial projection |
| 5 | `05_selective_scan_mamba.py` | Mamba's selective scan — input-dependent B, C, delta that assign higher update rates to important tokens |
| 6 | `06_parallel_scan.py` | Parallel prefix scan vs sequential scan — O(log L) depth using the associative recurrence operator |
| 7 | `07_char_level_ssm.py` | Minimal character-level language model using a selective SSM layer with fixed-size hidden state |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python 01_continuous_ssm.py
python 02_discretization_zoh.py
python 03_convolution_kernel.py
python 04_hippo_memory.py
python 05_selective_scan_mamba.py
python 06_parallel_scan.py
python 07_char_level_ssm.py
```
