"""CLIP training loop with synthetic data demo.

From: https://dadops.dev/blog/clip-from-scratch/
"""
import torch
import torch.nn.functional as F
from clip_model import CLIP
from clip_loss import clip_loss


def train_clip(model, dataloader, epochs=10, lr=3e-4):
    """Minimal CLIP training loop."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

    for epoch in range(epochs):
        total_loss = 0
        for images, token_ids in dataloader:
            # Forward pass: encode both modalities
            image_embeds, text_embeds, logit_scale = model(images, token_ids)

            # Compute symmetric contrastive loss
            loss = clip_loss(image_embeds, text_embeds, logit_scale)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Clamp logit_scale to prevent instability (after the update)
            with torch.no_grad():
                model.logit_scale.clamp_(max=torch.log(torch.tensor(100.0)))
            total_loss += loss.item()

        avg = total_loss / len(dataloader)
        temp = model.logit_scale.exp().item()
        print(f"Epoch {epoch+1}: loss={avg:.4f}, temperature=1/{1/temp:.4f}")


if __name__ == "__main__":
    # Train on synthetic data to verify the loop works
    torch.manual_seed(42)

    # Create a tiny model for testing
    model = CLIP(embed_dim=64, vision_dim=96, text_dim=64)

    # Synthetic dataset: random images and token IDs
    N = 32  # dataset size
    batch_size = 8
    images = torch.randn(N, 3, 224, 224)
    token_ids = torch.randint(0, 49152, (N, 10))

    # Simple dataloader using zip of chunks
    dataset = list(zip(
        images.split(batch_size),
        token_ids.split(batch_size)
    ))

    print(f"Training CLIP on {N} synthetic samples, batch_size={batch_size}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    train_clip(model, dataset, epochs=5, lr=1e-3)
    print("\nPASS")
