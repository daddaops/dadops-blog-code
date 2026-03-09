# Flow Matching from Scratch

Verified, runnable code from the [Flow Matching from Scratch](https://dadops.dev/blog/flow-matching-from-scratch/) blog post.

## Scripts

- **cfm_training.py** — Conditional Flow Matching training step
- **ode_solvers.py** — Euler, midpoint, and RK4 ODE samplers
- **reflow.py** — Reflow procedure and path straightness measurement
- **swiss_roll_training.py** — Complete Swiss roll training example
- **reflow_training.py** — Reflow applied to trained model

## Run

```bash
pip install -r requirements.txt
python cfm_training.py
python ode_solvers.py
python reflow.py
python swiss_roll_training.py
python reflow_training.py
```
