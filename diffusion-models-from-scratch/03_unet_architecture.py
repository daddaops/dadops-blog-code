import numpy as np

def sinusoidal_embedding(t, dim=64):
    """Sinusoidal timestep embedding -- same math as positional encoding.

    t: scalar timestep (or array of timesteps)
    dim: embedding dimension (must be even)
    """
    t = np.atleast_1d(np.array(t, dtype=np.float64))
    half = dim // 2
    # Frequency bands: 1/10000^(2i/dim)
    freqs = np.exp(-np.log(10000.0) * np.arange(half) / half)
    args = t[:, None] * freqs[None, :]
    return np.concatenate([np.sin(args), np.cos(args)], axis=-1)

class ResBlock:
    """Residual block with GroupNorm, SiLU, and timestep conditioning."""

    def __init__(self, channels, t_dim=64):
        # In a real framework: conv layers, norm, time projection
        self.channels = channels
        self.t_dim = t_dim
        # conv1: channels -> channels (3x3, padding=1)
        # conv2: channels -> channels (3x3, padding=1)
        # time_proj: t_dim -> channels (linear)
        # norm1, norm2: GroupNorm(num_groups=8, channels)

    def forward(self, x, t_emb):
        """x: [B, C, H, W], t_emb: [B, t_dim]"""
        residual = x
        # Block 1: norm -> activate -> convolve
        # h = group_norm(x)
        # h = silu(h)           # SiLU = x * sigmoid(x)
        # h = conv1(h)
        # Inject timestep: project and add
        # h = h + time_proj(t_emb)[:, :, None, None]
        # Block 2: norm -> activate -> convolve
        # h = group_norm(h)
        # h = silu(h)
        # h = conv2(h)
        # return h + residual   # skip connection
        return residual  # placeholder

class SimpleUNet:
    """Minimal U-Net for denoising 28x28 grayscale images.

    Architecture:
        Down: 1->32 -> 32->64 -> 64->128
        Bottleneck: 128->128
        Up: 128->64 -> 64->32 -> 32->1
    Each level has a ResBlock with timestep conditioning.
    Skip connections bridge down/up at each level.
    """

    def __init__(self, T=1000, t_dim=64):
        self.t_dim = t_dim
        # Timestep embedding: scalar -> t_dim vector
        # Down path (contracting)
        self.down1 = ResBlock(32, t_dim)   # 28x28
        self.down2 = ResBlock(64, t_dim)   # 14x14
        self.down3 = ResBlock(128, t_dim)  # 7x7
        # Bottleneck
        self.bottleneck = ResBlock(128, t_dim)
        # Up path (expanding) -- doubled channels from skip connections
        self.up3 = ResBlock(64, t_dim)     # 7x7 -> 14x14
        self.up2 = ResBlock(32, t_dim)     # 14x14 -> 28x28
        self.up1 = ResBlock(32, t_dim)     # 28x28
        # Final: channels -> 1 (predict noise, same shape as input)

    def forward(self, x, t):
        """x: [B, 1, 28, 28] noisy image, t: [B] timestep indices."""
        t_emb = sinusoidal_embedding(t, self.t_dim)

        # Encode (contract)
        # d1 = self.down1.forward(x, t_emb)       # 28x28, 32ch
        # d2 = self.down2.forward(downsample(d1), t_emb)  # 14x14, 64ch
        # d3 = self.down3.forward(downsample(d2), t_emb)  # 7x7, 128ch

        # Bottleneck
        # b = self.bottleneck.forward(d3, t_emb)   # 7x7, 128ch

        # Decode (expand) with skip connections
        # u3 = self.up3.forward(concat(upsample(b), d3), t_emb)   # 7x7
        # u2 = self.up2.forward(concat(upsample(u3), d2), t_emb)  # 14x14
        # u1 = self.up1.forward(concat(upsample(u2), d1), t_emb)  # 28x28

        # return final_conv(u1)  # [B, 1, 28, 28] noise prediction
        return x  # placeholder


# --- Demo: verify sinusoidal embedding and architecture ---
print("Sinusoidal embedding demo:")
for t in [0, 100, 500, 999]:
    emb = sinusoidal_embedding(t, dim=64)
    print(f"  t={t:4d}: shape={emb.shape}, "
          f"norm={np.linalg.norm(emb):.3f}, "
          f"first 4 values: [{emb[0,0]:.3f}, {emb[0,1]:.3f}, {emb[0,2]:.3f}, {emb[0,3]:.3f}]")

print("\nU-Net structure:")
unet = SimpleUNet(T=1000, t_dim=64)
print(f"  Down path channels: {unet.down1.channels} -> {unet.down2.channels} -> {unet.down3.channels}")
print(f"  Bottleneck channels: {unet.bottleneck.channels}")
print(f"  Up path channels: {unet.up3.channels} -> {unet.up2.channels} -> {unet.up1.channels}")
