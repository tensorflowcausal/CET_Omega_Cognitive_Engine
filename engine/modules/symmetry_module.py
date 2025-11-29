import torch
import torch.nn as nn


class SymmetryModule(nn.Module):
    """Toy symmetry learner.

    Attempts to find invariant directions in latent space by learning
    a linear transformation whose norm is regularized.
    This is a placeholder for more sophisticated symmetry discovery.
    """

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.transform = nn.Linear(latent_dim, latent_dim, bias=False)

    def forward(self, z: torch.Tensor):
        return self.transform(z)

    def regularization_loss(self) -> torch.Tensor:
        return torch.norm(self.transform.weight, p=2)
