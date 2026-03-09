# Neuroevolution from Scratch

Verified, runnable code from the [DadOps blog post](https://dadops.co/blog/neuroevolution-from-scratch/).

## Scripts

| Script | Description |
|--------|-------------|
| `ga_cartpole.py` | Simple weight-based GA for CartPole |
| `neat_genome.py` | NEAT genome representation (ConnectionGene, Genome, crossover) |
| `neat_xor.py` | NEAT XOR evolution with topological discovery |
| `cppn.py` | CPPN indirect encoding for HyperNEAT |
| `es_cartpole.py` | OpenAI Evolution Strategies for CartPole |

## Usage

```bash
python ga_cartpole.py      # GA weight evolution for CartPole
python neat_genome.py      # NEAT data structures demo
python neat_xor.py         # NEAT XOR evolution
python cppn.py             # CPPN weight matrix generation
python es_cartpole.py      # OpenAI ES for CartPole
```

Dependencies: numpy.
