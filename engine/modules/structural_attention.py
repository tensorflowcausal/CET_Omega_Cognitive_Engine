import torch
import torch.nn as nn
import torch.nn.functional as F


class StructuralAttention(nn.Module):
    """Structural attention A(z, psi).

    Takes a latent vector z and a scalar psi and returns:
      - z_mod: gated latent
      - gate: gating vector
    """

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc_gate = nn.Linear(latent_dim + 1, latent_dim)

    def forward(self, z: torch.Tensor, psi: torch.Tensor):
        if z.dim() != 2:
            z = z.view(z.size(0), -1)

        if psi.dim() == 0:
            psi = psi.view(1).expand(z.size(0))
        if psi.dim() == 1:
            psi = psi.unsqueeze(1)

        x = torch.cat([z, psi], dim=1)
        gate = torch.sigmoid(self.fc_gate(x))
        z_mod = gate * z
        return z_mod, gate
