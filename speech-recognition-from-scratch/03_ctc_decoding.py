"""CTC decoding: greedy vs beam search with prefix merging."""
import numpy as np


def greedy_decode(log_probs, alphabet, blank=0):
    """Greedy CTC decoding: argmax at each frame, then collapse."""
    path = np.argmax(log_probs, axis=1)
    # Collapse: merge duplicates, remove blanks
    result = []
    prev = -1
    for idx in path:
        if idx != prev:
            if idx != blank:
                result.append(alphabet[idx])
            prev = idx
    return ''.join(result)


def beam_decode(log_probs, alphabet, blank=0, beam_width=5):
    """Beam search CTC decoding with prefix merging."""
    T, C = log_probs.shape
    # Each beam: (prefix_string, log_prob_blank_end, log_prob_nonblank_end)
    LOG_ZERO = -1e30
    beams = {'': (0.0, LOG_ZERO)}  # start with empty prefix

    for t in range(T):
        new_beams = {}
        for prefix, (pb, pnb) in beams.items():
            p_total = np.logaddexp(pb, pnb)
            for c in range(C):
                lp = log_probs[t, c]
                if c == blank:
                    # Blank extends without changing prefix
                    key = prefix
                    new_pb = p_total + lp
                    if key in new_beams:
                        new_beams[key] = (np.logaddexp(new_beams[key][0], new_pb),
                                          new_beams[key][1])
                    else:
                        new_beams[key] = (new_pb, LOG_ZERO)
                else:
                    ch = alphabet[c]
                    # If repeating last char, only extend from blank-ending paths
                    if prefix and ch == prefix[-1]:
                        new_pnb = pb + lp  # must come via blank
                    else:
                        new_pnb = p_total + lp
                    key = prefix + ch
                    if key in new_beams:
                        new_beams[key] = (new_beams[key][0],
                                          np.logaddexp(new_beams[key][1], new_pnb))
                    else:
                        new_beams[key] = (LOG_ZERO, new_pnb)
        # Prune to top beam_width
        scored = {k: np.logaddexp(v[0], v[1]) for k, v in new_beams.items()}
        top_keys = sorted(scored, key=scored.get, reverse=True)[:beam_width]
        beams = {k: new_beams[k] for k in top_keys}

    best = max(beams, key=lambda k: np.logaddexp(beams[k][0], beams[k][1]))
    return best


if __name__ == "__main__":
    # Synthetic example where greedy fails but beam search succeeds
    T = 10
    alphabet = ['_', 'g', 'o']
    probs = np.array([
        [0.05, 0.90, 0.05],  # strong g
        [0.60, 0.30, 0.10],  # g fading
        [0.90, 0.03, 0.07],  # blank transition
        [0.51, 0.01, 0.48],  # blank BARELY beats o
        [0.51, 0.01, 0.48],  # blank BARELY beats o
        [0.51, 0.01, 0.48],  # blank BARELY beats o
        [0.51, 0.01, 0.48],  # blank BARELY beats o
        [0.95, 0.01, 0.04],  # strong blank
        [0.97, 0.01, 0.02],  # strong blank
        [0.99, 0.005, 0.005] # strong blank
    ])
    log_probs = np.log(probs)
    print(f"Greedy:     '{greedy_decode(log_probs, alphabet)}'")
    print(f"Beam (B=5): '{beam_decode(log_probs, alphabet, beam_width=5)}'")
