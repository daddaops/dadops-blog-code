# Spiking Neural Networks from Scratch

Code extracted from the [DadOps blog post](https://dadops.co/blog/spiking-neural-networks-from-scratch/).

## Scripts

1. **01_lif_neuron.py** - Leaky Integrate-and-Fire neuron simulation with constant current injection
2. **02_spike_encoding.py** - Rate coding and temporal coding schemes for converting analog values to spike trains
3. **03_spiking_forward.py** - Two-layer spiking neural network forward pass (input -> hidden -> output)
4. **04_stdp_learning.py** - Spike-Timing-Dependent Plasticity (STDP) learning rule with causal and anti-causal pairings
5. **05_surrogate_gradient.py** - Surrogate gradient method (fast sigmoid) for training SNNs via backpropagation
6. **06_full_snn_pipeline.py** - Complete SNN training pipeline: rate encoding, surrogate gradient training, and evaluation on a 3-class classification task

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python 01_lif_neuron.py
python 02_spike_encoding.py
python 03_spiking_forward.py
python 04_stdp_learning.py
python 05_surrogate_gradient.py
python 06_full_snn_pipeline.py
```
