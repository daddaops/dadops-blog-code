import torch
import torch.nn as nn
from torch.autograd import Function

class GradientReversal(Function):
    """Gradient Reversal Layer: identity forward, negate backward."""

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

class DANN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Task classifier (source labels)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        # Domain discriminator (source vs target)
        self.domain_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, lambd=1.0):
        feats = self.features(x)
        class_logits = self.classifier(feats)
        # Reverse gradients before domain head
        reversed_feats = GradientReversal.apply(feats, lambd)
        domain_logits = self.domain_head(reversed_feats)
        return class_logits, domain_logits

# Lambda schedule: sigmoid warmup from 0 to 1
def dann_lambda(progress):
    """progress in [0, 1]: fraction of training completed."""
    return 2.0 / (1.0 + torch.exp(torch.tensor(-10.0 * progress))) - 1.0

# --- Quick smoke test ---
model = DANN(input_dim=20, hidden_dim=32, num_classes=3)
x = torch.randn(8, 20)
class_out, domain_out = model(x, lambd=1.0)
print(f"Class logits shape: {class_out.shape}")   # [8, 3]
print(f"Domain logits shape: {domain_out.shape}")  # [8, 1]
print(f"Lambda at 50%: {dann_lambda(0.5):.4f}")
print(f"Lambda at 100%: {dann_lambda(1.0):.4f}")
print("DANN model OK")
