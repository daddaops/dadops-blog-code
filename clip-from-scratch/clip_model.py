"""CLIP model architecture: Vision encoder, Text encoder, and full CLIP model.

From: https://dadops.dev/blog/clip-from-scratch/
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionEncoder(nn.Module):
    """Simplified ViT image encoder for CLIP."""

    def __init__(self, image_size=224, patch_size=16, channels=3,
                 embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2

        # Patch embedding: project each patch to embed_dim
        self.patch_embed = nn.Conv2d(
            channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

        # Standard transformer encoder
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, images):
        # images: [B, C, H, W]
        x = self.patch_embed(images)        # [B, embed_dim, grid, grid]
        x = x.flatten(2).transpose(1, 2)    # [B, num_patches, embed_dim]

        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)      # [B, num_patches + 1, embed_dim]
        x = x + self.pos_embed

        x = self.transformer(x)
        return self.ln(x[:, 0])              # [CLS] token output


class TextEncoder(nn.Module):
    """Simplified text encoder for CLIP."""

    def __init__(self, vocab_size=49152, max_len=77,
                 embed_dim=512, depth=12, num_heads=8):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, token_ids):
        # token_ids: [B, seq_len]
        x = self.token_embed(token_ids) + self.pos_embed[:, :token_ids.size(1)]

        # Causal mask: each token can only attend to previous tokens
        mask = torch.triu(
            torch.ones(token_ids.size(1), token_ids.size(1), device=x.device),
            diagonal=1
        ).bool()

        x = self.transformer(x, mask=mask)

        # Extract the [EOS] token output (CLIP uses the highest-id token,
        # which is [EOS] at position 49407 in the vocabulary)
        eos_indices = token_ids.argmax(dim=-1)
        x = x[torch.arange(x.size(0)), eos_indices]
        return self.ln(x)


class CLIP(nn.Module):
    """Complete CLIP model: two encoders projecting to a shared space."""

    def __init__(self, embed_dim=512, vision_dim=768, text_dim=512):
        super().__init__()
        self.vision_encoder = VisionEncoder(embed_dim=vision_dim)
        self.text_encoder = TextEncoder(embed_dim=text_dim)

        # Linear projections to shared embedding space
        self.vision_proj = nn.Linear(vision_dim, embed_dim, bias=False)
        self.text_proj = nn.Linear(text_dim, embed_dim, bias=False)

        # Learned temperature (initialized to 1/0.07 ≈ 14.3)
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(
            torch.tensor(1.0 / 0.07)
        ))

    def encode_image(self, images):
        features = self.vision_encoder(images)
        projected = self.vision_proj(features)
        return F.normalize(projected, dim=-1)   # L2 normalize

    def encode_text(self, token_ids):
        features = self.text_encoder(token_ids)
        projected = self.text_proj(features)
        return F.normalize(projected, dim=-1)   # L2 normalize

    def forward(self, images, token_ids):
        image_embeds = self.encode_image(images)   # [B, embed_dim]
        text_embeds = self.encode_text(token_ids)   # [B, embed_dim]
        return image_embeds, text_embeds, self.logit_scale.exp()


if __name__ == "__main__":
    # Quick smoke test with tiny model
    model = CLIP(embed_dim=64, vision_dim=96, text_dim=64)
    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    print(f"CLIP model created with {total:,} parameters")

    # Test forward pass with dummy data
    images = torch.randn(2, 3, 224, 224)
    token_ids = torch.randint(0, 49152, (2, 10))
    image_embeds, text_embeds, logit_scale = model(images, token_ids)
    print(f"Image embeddings shape: {image_embeds.shape}")
    print(f"Text embeddings shape:  {text_embeds.shape}")
    print(f"Logit scale: {logit_scale.item():.2f}")

    # Verify L2 normalization
    norms = image_embeds.norm(dim=-1)
    print(f"Image embedding norms: {norms.tolist()} (should be ~1.0)")
    print("PASS")
