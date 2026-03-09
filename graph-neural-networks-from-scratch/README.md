# Graph Neural Networks from Scratch

Verified, runnable code from the [Graph Neural Networks from Scratch](https://dadops.dev/blog/graph-neural-networks-from-scratch/) blog post.

## Scripts

- **graph_setup.py** — Graph adjacency matrix, node features, degree matrix
- **message_passing.py** — Simple message passing layer
- **gcn.py** — Graph Convolutional Network (Kipf & Welling 2017)
- **graphsage.py** — GraphSAGE with mean aggregator and neighbor sampling
- **gat.py** — Graph Attention Network (Velickovic et al. 2018)
- **wl_test.py** — Weisfeiler-Leman color refinement (1-WL)
- **graph_pooling.py** — Graph readout/pooling for graph-level prediction

## Run

```bash
pip install -r requirements.txt
python graph_setup.py
python message_passing.py
python gcn.py
python graphsage.py
python gat.py
python wl_test.py
python graph_pooling.py
```
