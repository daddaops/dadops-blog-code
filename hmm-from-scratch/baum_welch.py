import numpy as np
from hmm import HMM, casino, forward, backward

def baum_welch(obs, N, M, max_iter=100, seed=0):
    """Learn HMM parameters from observations using EM."""
    rng = np.random.RandomState(seed)

    # Random initialization
    A = rng.dirichlet(np.ones(N), size=N)
    B = rng.dirichlet(np.ones(M), size=N)
    pi = rng.dirichlet(np.ones(N))
    hmm = HMM(A, B, pi)
    T = len(obs)

    for iteration in range(max_iter):
        # E-step: forward-backward
        log_alpha, log_prob = forward(hmm, obs)
        log_beta = backward(hmm, obs)

        # Compute gamma: P(X_t = i | O)
        log_gamma = log_alpha + log_beta - log_prob
        gamma = np.exp(log_gamma)

        # Compute xi: P(X_t = i, X_{t+1} = j | O)
        xi = np.zeros((T-1, N, N))
        for t in range(T-1):
            for i in range(N):
                for j in range(N):
                    xi[t, i, j] = np.exp(
                        log_alpha[t, i] + np.log(hmm.A[i, j])
                        + np.log(hmm.B[j, obs[t+1]]) + log_beta[t+1, j]
                        - log_prob
                    )

        # M-step: update parameters
        hmm.pi = gamma[0]
        for i in range(N):
            denom = gamma[:-1, i].sum()
            for j in range(N):
                hmm.A[i, j] = xi[:, i, j].sum() / denom
            for k in range(M):
                mask = (obs == k)
                hmm.B[i, k] = gamma[mask, i].sum() / gamma[:, i].sum()

        if iteration % 20 == 0:
            print(f"Iter {iteration:3d}: log P(O) = {log_prob:.2f}")

    print(f"Iter {iteration:3d}: log P(O) = {log_prob:.2f}")
    return hmm

if __name__ == "__main__":
    # Generate longer sequence for better learning
    _, long_obs = casino.generate(T=500, seed=99)
    learned = baum_welch(long_obs, N=2, M=6, max_iter=100)

    print(f"\nLearned transitions:\n{learned.A.round(3)}")
    print(f"True transitions:\n{casino.A}")
    print(f"\nLearned emissions:\n{learned.B.round(3)}")
    print(f"True emissions:\n{np.array(casino.B).round(3)}")
    # Learned parameters converge close to true values
