# Knowledge Graphs from Scratch

Verified, runnable code from the [Knowledge Graphs from Scratch](https://dadops.dev/blog/knowledge-graphs-from-scratch/) blog post.

## Scripts

- **graph_basics.py** — Knowledge graph as triples, adjacency traversal, one/two-hop queries
- **transe.py** — TransE embedding training with margin-based ranking loss
- **scoring.py** — DistMult (symmetric) and ComplEx (asymmetric) scoring functions
- **evaluation.py** — Link prediction evaluation (MR, MRR, Hits@k) with filtered ranking
- **multi_hop.py** — Multi-hop reasoning by composing TransE relation vectors
- **extraction.py** — Rule-based triple extraction from text + two-hop inference

## Run

```bash
pip install -r requirements.txt
python graph_basics.py
python transe.py
python scoring.py
python evaluation.py
python multi_hop.py
python extraction.py
```
