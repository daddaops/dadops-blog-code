import numpy as np

def score_distmult(h, r, t):
    """DistMult: element-wise product, then sum. Symmetric."""
    return np.sum(h * r * t)

def score_complex(h_re, h_im, r_re, r_im, t_re, t_im):
    """ComplEx: uses complex conjugate of tail. Asymmetric."""
    # Re(h * r * conj(t))
    return (np.sum(h_re * r_re * t_re) + np.sum(h_re * r_im * t_im)
          + np.sum(h_im * r_re * t_im) - np.sum(h_im * r_im * t_re))

rng = np.random.RandomState(7)
dim = 10

# DistMult: same score both directions
paris, france, capital = rng.randn(dim), rng.randn(dim), rng.randn(dim)
fwd = score_distmult(paris, capital, france)
bwd = score_distmult(france, capital, paris)
print(f"DistMult (Paris, capital_of, France): {fwd:.3f}")
print(f"DistMult (France, capital_of, Paris): {bwd:.3f}")
print(f"Same? {abs(fwd - bwd) < 1e-10}")  # True!

# ComplEx: different scores (correctly asymmetric)
p_re, p_im = rng.randn(dim), rng.randn(dim)
f_re, f_im = rng.randn(dim), rng.randn(dim)
c_re, c_im = rng.randn(dim), rng.randn(dim)
fwd = score_complex(p_re, p_im, c_re, c_im, f_re, f_im)
bwd = score_complex(f_re, f_im, c_re, c_im, p_re, p_im)
print(f"\nComplEx (Paris, capital_of, France): {fwd:.3f}")
print(f"ComplEx (France, capital_of, Paris): {bwd:.3f}")
print(f"Same? {abs(fwd - bwd) < 1e-10}")  # False!
