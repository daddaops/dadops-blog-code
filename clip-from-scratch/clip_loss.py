"""CLIP contrastive loss: symmetric cross-entropy over cosine similarities.

From: https://dadops.dev/blog/clip-from-scratch/
"""
import torch
import torch.nn.functional as F


def clip_loss(image_embeds, text_embeds, logit_scale):
    """Symmetric contrastive loss for CLIP.

    Args:
        image_embeds: [N, D] L2-normalized image embeddings
        text_embeds:  [N, D] L2-normalized text embeddings
        logit_scale:  scalar (exp of learned log-temperature)
    """
    # Cosine similarity matrix scaled by temperature
    logits = logit_scale * image_embeds @ text_embeds.T   # [N, N]

    # Labels: image_i matches text_i
    labels = torch.arange(len(image_embeds), device=logits.device)

    # Cross-entropy in both directions
    loss_i2t = F.cross_entropy(logits, labels)        # rows: image -> text
    loss_t2i = F.cross_entropy(logits.T, labels)      # cols: text -> image

    return (loss_i2t + loss_t2i) / 2


# ---- CLIP pseudocode (adapted from Radford et al., 2021, Figure 3) ----

# I_f = image_encoder(images)                  # [N, d_image]
# T_f = text_encoder(texts)                    # [N, d_text]

# Project to shared space and normalize
# I_e = l2_normalize(I_f @ W_image)            # [N, d_embed]
# T_e = l2_normalize(T_f @ W_text)             # [N, d_embed]

# Scaled pairwise cosine similarities
# logits = I_e @ T_e.T * exp(temperature)      # [N, N]

# Symmetric cross-entropy
# labels = arange(N)                           # [0, 1, 2, ..., N-1]
# loss_i = cross_entropy(logits, labels, axis=0)   # image-to-text
# loss_t = cross_entropy(logits, labels, axis=1)   # text-to-image
# loss   = (loss_i + loss_t) / 2


if __name__ == "__main__":
    # Demo with synthetic embeddings
    torch.manual_seed(42)
    N, D = 8, 64

    # Create synthetic "matched" embeddings (similar on diagonal)
    base = torch.randn(N, D)
    image_embeds = F.normalize(base + torch.randn(N, D) * 0.1, dim=-1)
    text_embeds = F.normalize(base + torch.randn(N, D) * 0.1, dim=-1)

    logit_scale = torch.tensor(14.3)  # 1/0.07

    loss = clip_loss(image_embeds, text_embeds, logit_scale)
    print(f"Batch size: {N}")
    print(f"Embedding dim: {D}")
    print(f"Logit scale: {logit_scale.item():.1f}")
    print(f"Contrastive loss: {loss.item():.4f}")

    # Show the similarity matrix
    sims = (image_embeds @ text_embeds.T).detach()
    print(f"\nCosine similarity matrix (should be high on diagonal):")
    for i in range(N):
        row = " ".join(f"{sims[i,j]:+.2f}" for j in range(N))
        print(f"  [{row}]")

    # Test with random (unmatched) embeddings — loss should be higher
    random_text = F.normalize(torch.randn(N, D), dim=-1)
    loss_random = clip_loss(image_embeds, random_text, logit_scale)
    print(f"\nLoss with matched pairs:  {loss.item():.4f}")
    print(f"Loss with random pairs:   {loss_random.item():.4f}")
    print(f"Matched < Random: {loss.item() < loss_random.item()}")
    print("PASS")
