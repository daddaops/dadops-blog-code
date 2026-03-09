# Program Synthesis from Scratch

Verified, runnable code from the [DadOps blog post](https://dadops.co/blog/program-synthesis-from-scratch/).

## Scripts

| Script | Description |
|--------|-------------|
| `enumerative_synthesis.py` | Bottom-up enumeration with equivalence pruning |
| `cegis_synthesis.py` | Counter-Example Guided Inductive Synthesis |
| `string_synthesis.py` | FlashFill-style string transformation |
| `neural_guided_synthesis.py` | MLP-guided operation prediction |
| `llm_synthesis_repair_loop.py` | LLM self-repair synthesis loop |

## Usage

```bash
python enumerative_synthesis.py      # Find f(x) = 2x+1
python cegis_synthesis.py            # CEGIS for f(x) = x²+1
python string_synthesis.py           # FlashFill name formatting
python neural_guided_synthesis.py    # Neural op prediction
python llm_synthesis_repair_loop.py  # LLM self-repair demo
```

Dependencies: numpy (for neural_guided_synthesis.py only).
