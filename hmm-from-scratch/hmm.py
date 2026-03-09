import numpy as np

class HMM:
    def __init__(self, A, B, pi):
        self.A = np.array(A, dtype=float)   # N x N transition
        self.B = np.array(B, dtype=float)   # N x M emission
        self.pi = np.array(pi, dtype=float) # N initial probs
        self.N = self.A.shape[0]            # num hidden states
        self.M = self.B.shape[1]            # num observation symbols

    def generate(self, T, seed=42):
        rng = np.random.RandomState(seed)
        states, obs = [], []
        state = rng.choice(self.N, p=self.pi)
        for _ in range(T):
            states.append(state)
            obs.append(rng.choice(self.M, p=self.B[state]))
            state = rng.choice(self.N, p=self.A[state])
        return np.array(states), np.array(obs)

def forward(hmm, obs):
    """Forward algorithm in log-space. Returns log(alpha) trellis and log P(O|model)."""
    T = len(obs)
    log_alpha = np.full((T, hmm.N), -np.inf)

    # Base case: t=0
    for i in range(hmm.N):
        log_alpha[0, i] = np.log(hmm.pi[i]) + np.log(hmm.B[i, obs[0]])

    # Recursion: t=1..T-1
    for t in range(1, T):
        for j in range(hmm.N):
            log_sum = np.logaddexp.reduce(
                log_alpha[t-1, :] + np.log(hmm.A[:, j])
            )
            log_alpha[t, j] = log_sum + np.log(hmm.B[j, obs[t]])

    log_prob = np.logaddexp.reduce(log_alpha[T-1, :])
    return log_alpha, log_prob

def viterbi(hmm, obs):
    """Viterbi algorithm in log-space. Returns best state sequence and its log-prob."""
    T = len(obs)
    log_delta = np.full((T, hmm.N), -np.inf)
    psi = np.zeros((T, hmm.N), dtype=int)

    # Base case
    for i in range(hmm.N):
        log_delta[0, i] = np.log(hmm.pi[i]) + np.log(hmm.B[i, obs[0]])

    # Recursion
    for t in range(1, T):
        for j in range(hmm.N):
            scores = log_delta[t-1, :] + np.log(hmm.A[:, j])
            psi[t, j] = np.argmax(scores)
            log_delta[t, j] = scores[psi[t, j]] + np.log(hmm.B[j, obs[t]])

    # Backtrack
    path = np.zeros(T, dtype=int)
    path[T-1] = np.argmax(log_delta[T-1, :])
    for t in range(T-2, -1, -1):
        path[t] = psi[t+1, path[t+1]]

    return path, np.max(log_delta[T-1, :])

def backward(hmm, obs):
    """Backward algorithm in log-space. Returns log(beta) trellis."""
    T = len(obs)
    log_beta = np.full((T, hmm.N), -np.inf)
    log_beta[T-1, :] = 0.0  # log(1) = 0

    for t in range(T-2, -1, -1):
        for i in range(hmm.N):
            log_beta[t, i] = np.logaddexp.reduce(
                np.log(hmm.A[i, :]) + np.log(hmm.B[:, obs[t+1]]) + log_beta[t+1, :]
            )
    return log_beta

def posterior_decode(hmm, obs):
    """Compute posterior state probabilities and decode."""
    log_alpha, log_prob = forward(hmm, obs)
    log_beta = backward(hmm, obs)

    # gamma[t, i] = P(X_t = i | O, model)
    log_gamma = log_alpha + log_beta - log_prob
    gamma = np.exp(log_gamma)

    # Posterior decoding: pick most likely state at each t
    path = np.argmax(gamma, axis=1)
    return gamma, path

# Dishonest casino: state 0 = Fair, state 1 = Loaded
casino = HMM(
    A=[[0.95, 0.05],   # Fair stays Fair 95%, switches to Loaded 5%
       [0.10, 0.90]],  # Loaded stays Loaded 90%, switches to Fair 10%
    B=[[1/6]*6,                          # Fair die: uniform
       [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]], # Loaded die: 6 has prob 0.5
    pi=[0.5, 0.5]
)

if __name__ == "__main__":
    true_states, observations = casino.generate(T=30)
    faces = observations + 1
    print("Rolls: ", " ".join(str(f) for f in faces))
    print("Truth: ", " ".join("F" if s == 0 else "L" for s in true_states))
