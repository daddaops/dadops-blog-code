# Sparse Autoencoders from Scratch

Code extracted from the blog post: [Sparse Autoencoders from Scratch](https://dadops.substack.com/sparse-autoencoders-from-scratch)

## Scripts

1. **basic_autoencoder.py** — Demonstrates the superposition problem: 5 sparse features packed into 2 dimensions, showing polysemantic neurons
2. **sparse_autoencoder.py** — The SparseAutoencoder class: overcomplete autoencoder with ReLU activation and decoder norm constraint
3. **loss_functions.py** — SAE loss function (reconstruction MSE + L1 sparsity penalty) and L1 gradient
4. **train_synthetic.py** — Full training loop on synthetic data with known ground-truth features, verifying feature recovery via cosine similarity
5. **monosemanticity.py** — Trains an MLP classifier, then decomposes its hidden layer with an SAE to show polysemantic neurons becoming monosemantic features
6. **topk_sae.py** — Top-K sparse autoencoder variant (Gao et al., 2024): sparsity by selection instead of L1 penalization

## Usage

```bash
pip install -r requirements.txt
python basic_autoencoder.py
python sparse_autoencoder.py
python loss_functions.py
python train_synthetic.py
python monosemanticity.py
python topk_sae.py
```

Each script is standalone and runnable. Scripts 4 and 5 include their own copies of the SparseAutoencoder class and loss functions so they can run independently.
