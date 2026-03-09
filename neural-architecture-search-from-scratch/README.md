# Neural Architecture Search from Scratch

Verified, runnable code from the [DadOps blog post](https://dadops.co/blog/neural-architecture-search-from-scratch/).

## Scripts

| Script | Description |
|--------|-------------|
| `search_space.py` | Cell-based search space with 5.6B architectures |
| `successive_halving.py` | Random search with successive halving |
| `evolutionary_search.py` | Regularized evolution with mutation and tournament selection |
| `darts_search.py` | DARTS differentiable architecture search |
| `supernet.py` | One-shot NAS with shared weight supernet |
| `hardware_aware.py` | Hardware-aware NAS with accuracy/latency trade-off |

## Usage

```bash
python search_space.py         # Search space size calculation
python successive_halving.py   # Successive halving demo
python evolutionary_search.py  # Evolutionary search demo
python darts_search.py         # DARTS bilevel optimization
python supernet.py             # Supernet weight sharing
python hardware_aware.py       # Hardware-aware multi-objective
```

Dependencies: numpy.
