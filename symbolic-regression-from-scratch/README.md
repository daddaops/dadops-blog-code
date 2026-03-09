# Symbolic Regression from Scratch

Code extracted from the [DadOps blog post](https://dadops.dev/blog/symbolic-regression-from-scratch/).

## Scripts

| # | Script | Description |
|---|--------|-------------|
| 1 | `01_expression_tree.py` | Expression tree `Node` class with evaluate, size, depth, and string representation |
| 2 | `02_tree_generation.py` | Random tree generation with grow method and Ramped Half-and-Half initialization |
| 3 | `03_genetic_operators.py` | Tournament selection, subtree crossover, point mutation, and hoist mutation |
| 4 | `04_fitness_evaluation.py` | MSE + parsimony pressure fitness and Pareto front computation |
| 5 | `05_linear_scaling.py` | Linear scaling fitness: learns optimal `a * f(x) + b` via closed-form least squares |
| 6 | `06_symbolic_regression.py` | Full GP loop — evolves population for 50 generations to discover `f(x) = x^2 + sin(x)` |

## Shared Modules

- `node.py` — `Node` class (expression tree nodes)
- `tree_gen.py` — Tree generation utilities
- `genetic_ops.py` — Genetic operators
- `linear_scaling.py` — Linear scaling fitness function

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python 01_expression_tree.py
python 02_tree_generation.py
python 03_genetic_operators.py
python 04_fitness_evaluation.py
python 05_linear_scaling.py
python 06_symbolic_regression.py
```

## Output

Pre-generated output from each script is saved in the `output/` directory.
