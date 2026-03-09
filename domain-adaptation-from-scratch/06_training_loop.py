import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.data import DataLoader, TensorDataset

# --- Reuse from block 4 ---
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
        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.domain_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, lambd=1.0):
        feats = self.features(x)
        class_logits = self.classifier(feats)
        reversed_feats = GradientReversal.apply(feats, lambd)
        domain_logits = self.domain_head(reversed_feats)
        return class_logits, domain_logits
# --- End reuse from block 4 ---

def train_adapted_model(model, source_loader, target_loader,
                        num_epochs=50, lr=0.001):
    """Full domain adaptation training loop with CORAL + DANN."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    class_loss_fn = nn.CrossEntropyLoss()
    domain_loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        progress = epoch / num_epochs
        lambd = 2.0 / (1.0 + float(torch.exp(torch.tensor(-10.0 * progress)))) - 1.0

        for (src_x, src_y), (tgt_x, _) in zip(source_loader, target_loader):
            # Forward pass through shared feature extractor
            src_feats = model.features(src_x)
            tgt_feats = model.features(tgt_x)

            # Task loss: classification on source only
            class_logits = model.classifier(src_feats)
            loss_class = class_loss_fn(class_logits, src_y)

            # CORAL loss: align covariance matrices
            src_centered = src_feats - src_feats.mean(dim=0)
            tgt_centered = tgt_feats - tgt_feats.mean(dim=0)
            cov_s = (src_centered.T @ src_centered) / (len(src_x) - 1)
            cov_t = (tgt_centered.T @ tgt_centered) / (len(tgt_x) - 1)
            d = src_feats.shape[1]
            loss_coral = torch.sum((cov_s - cov_t) ** 2) / (4 * d * d)

            # Domain loss: adversarial with gradient reversal
            src_domain = model.domain_head(
                GradientReversal.apply(src_feats, lambd))
            tgt_domain = model.domain_head(
                GradientReversal.apply(tgt_feats, lambd))
            labels_src = torch.zeros(len(src_x), 1)
            labels_tgt = torch.ones(len(tgt_x), 1)
            loss_domain = (domain_loss_fn(src_domain, labels_src) +
                           domain_loss_fn(tgt_domain, labels_tgt))

            # Combined loss
            loss = loss_class + 0.1 * loss_coral + loss_domain
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: class={loss_class:.3f} "
                  f"coral={loss_coral:.3f} domain={loss_domain:.3f} "
                  f"lambda={lambd:.2f}")

# --- Demo with synthetic data ---
torch.manual_seed(42)
input_dim, hidden_dim, num_classes = 20, 32, 3
n_samples = 200

# Source data: class-separable
src_x = torch.randn(n_samples, input_dim)
src_y = torch.randint(0, num_classes, (n_samples,))

# Target data: shifted distribution (no labels used in training)
tgt_x = torch.randn(n_samples, input_dim) + 0.5
tgt_y = torch.zeros(n_samples, dtype=torch.long)  # placeholder

source_loader = DataLoader(TensorDataset(src_x, src_y), batch_size=64, shuffle=True)
target_loader = DataLoader(TensorDataset(tgt_x, tgt_y), batch_size=64, shuffle=True)

model = DANN(input_dim, hidden_dim, num_classes)
train_adapted_model(model, source_loader, target_loader, num_epochs=50, lr=0.001)
print("Training complete.")
