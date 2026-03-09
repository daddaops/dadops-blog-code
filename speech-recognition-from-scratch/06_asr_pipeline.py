"""End-to-end ASR pipeline simulation: spectrogram -> CTC decode -> WER."""
import numpy as np


def simulate_asr_pipeline(utterance, n_frames=15):
    """End-to-end ASR simulation: spectrogram -> CTC logits -> decode -> WER."""
    np.random.seed(42)
    chars = list("_abcdefghijklmnopqrstuvwxyz ")  # blank=0, space=27
    char2idx = {c: i for i, c in enumerate(chars)}
    C = len(chars)

    # Step 1: Simulate model output (frame-level logits)
    logits = np.random.randn(n_frames, C) * 0.3
    # Bias toward correct characters at appropriate frames
    target_chars = list(utterance)
    frames_per_char = n_frames / len(target_chars)
    for ci, ch in enumerate(target_chars):
        t_start = int(ci * frames_per_char)
        t_end = int((ci + 1) * frames_per_char)
        idx = char2idx.get(ch, 0)
        for t in range(t_start, min(t_end, n_frames)):
            logits[t, idx] += 2.5

    # Step 2: CTC greedy decode
    log_probs = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
    path = np.argmax(log_probs, axis=1)
    result = []
    prev = -1
    for idx in path:
        if idx != prev:
            if idx != 0:  # skip blank
                result.append(chars[idx])
            prev = idx
    hypothesis = ''.join(result)

    # Step 3: Compute WER
    ref_words = utterance.split()
    hyp_words = hypothesis.split()
    N, M = len(ref_words), len(hyp_words)
    dp = np.zeros((N+1, M+1), dtype=int)
    for i in range(N+1): dp[i, 0] = i
    for j in range(M+1): dp[0, j] = j
    for i in range(1, N+1):
        for j in range(1, M+1):
            cost = 0 if ref_words[i-1] == hyp_words[j-1] else 1
            dp[i,j] = min(dp[i-1,j-1]+cost, dp[i-1,j]+1, dp[i,j-1]+1)
    wer = dp[N, M] / N if N > 0 else 0.0

    print(f"Utterance:  '{utterance}'")
    print(f"Frames:      {n_frames} (logits shape: {logits.shape})")
    print(f"Decoded:    '{hypothesis}'")
    print(f"WER:         {wer:.1%}")
    return hypothesis, wer


if __name__ == "__main__":
    simulate_asr_pipeline("hi dad", n_frames=15)
