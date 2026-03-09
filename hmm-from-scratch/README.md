# Hidden Markov Models from Scratch

Verified, runnable code from the [HMM from Scratch](https://dadops.dev/blog/hmm-from-scratch/) blog post.

## Scripts

- **markov_chain.py** — Markov chain simulation and stationary distribution
- **hmm.py** — HMM class, forward/backward/Viterbi algorithms, dishonest casino model
- **forward_algorithm.py** — Forward algorithm demo: sequence probability
- **viterbi_algorithm.py** — Viterbi decoding: most likely state sequence
- **posterior_decoding.py** — Posterior decoding: marginal state probabilities
- **baum_welch.py** — Baum-Welch EM algorithm: learn HMM parameters from data

## Run

```bash
pip install -r requirements.txt
python markov_chain.py
python hmm.py
python forward_algorithm.py
python viterbi_algorithm.py
python posterior_decoding.py
python baum_welch.py
```
