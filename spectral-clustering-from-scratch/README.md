# Spectral Clustering from Scratch

Standalone Python scripts extracted from the DadOps blog post on spectral clustering.

## Scripts

| Script | Description |
|--------|-------------|
| `01_similarity_graphs.py` | Build three similarity graph types (fully connected RBF, k-NN, epsilon-neighborhood) from two-moons data |
| `02_graph_laplacian.py` | Compute graph Laplacian and examine eigenvalue spectrum to detect clusters via eigengap |
| `03_spectral_clustering.py` | Full spectral clustering pipeline (NJW algorithm) compared with k-means on two moons |
| `04_visualize_pipeline.py` | Visualize original data, eigenvector embedding, and clustering result (saves PNG to `output/`) |
| `05_graph_cuts.py` | Compare Cut, RatioCut, and Ncut metrics between spectral and random partitions |
| `06_eigengap_heuristic.py` | Use the eigengap heuristic to estimate number of clusters on moons, blobs, and circles datasets |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python 01_similarity_graphs.py
python 02_graph_laplacian.py
python 03_spectral_clustering.py
python 04_visualize_pipeline.py
python 05_graph_cuts.py
python 06_eigengap_heuristic.py
```
