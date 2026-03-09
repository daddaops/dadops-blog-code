import numpy as np


def clip_contrastive_loss(image_emb, text_emb, temperature=0.07):
    """
    Symmetric contrastive loss for image-text pairs.
    image_emb: (N, D) L2-normalized image embeddings
    text_emb:  (N, D) L2-normalized text embeddings
    """
    # Similarity matrix: each image against each text caption
    logits = image_emb @ text_emb.T / temperature  # (N, N)

    # Labels: the diagonal — image_i matches text_i
    N = logits.shape[0]
    labels = np.arange(N)

    # Image-to-text: for each image, which text is correct?
    i2t_loss = softmax_cross_entropy(logits, labels)

    # Text-to-image: for each text, which image is correct?
    t2i_loss = softmax_cross_entropy(logits.T, labels)

    return (i2t_loss + t2i_loss) / 2


def softmax_cross_entropy(logits, labels):
    """Numerically stable cross-entropy with log-sum-exp trick."""
    N = logits.shape[0]
    max_logits = logits.max(axis=1, keepdims=True)
    log_sum_exp = max_logits.squeeze() + np.log(
        np.sum(np.exp(logits - max_logits), axis=1)
    )
    correct_logits = logits[np.arange(N), labels]
    return np.mean(log_sum_exp - correct_logits)


if __name__ == "__main__":
    np.random.seed(42)
    N, D = 8, 16

    # Random embeddings (L2-normalized)
    img_raw = np.random.randn(N, D)
    image_emb = img_raw / np.linalg.norm(img_raw, axis=1, keepdims=True)
    txt_raw = np.random.randn(N, D)
    text_emb = txt_raw / np.linalg.norm(txt_raw, axis=1, keepdims=True)

    loss_random = clip_contrastive_loss(image_emb, text_emb)
    print(f"CLIP loss (random embeddings):  {loss_random:.4f}")

    # Matched embeddings (image_i == text_i) — loss should be lower
    text_emb_matched = image_emb.copy() + np.random.randn(N, D) * 0.05
    text_emb_matched = text_emb_matched / np.linalg.norm(text_emb_matched, axis=1, keepdims=True)
    loss_matched = clip_contrastive_loss(image_emb, text_emb_matched)
    print(f"CLIP loss (near-matched pairs): {loss_matched:.4f}")

    # Perfect match
    loss_perfect = clip_contrastive_loss(image_emb, image_emb)
    print(f"CLIP loss (perfect match):      {loss_perfect:.4f}")
    print(f"\nLoss decreases as alignment improves: {loss_random > loss_matched > loss_perfect}")
