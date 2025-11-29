import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.encoder import Encoder
from .modules.structural_attention import StructuralAttention
from .modules.world_model import WorldModel
from .modules.symmetry_module import SymmetryModule
from .modules.psi_update import update_psi


class CETOmegaCognitiveEngine(nn.Module):
    """CET Ω Cognitive Engine (non-agentic, representational).

    Components:
      - Encoder E(x) -> z
      - Structural attention A(z, psi) -> z_mod
      - World model M(z_mod) -> x_pred
      - Symmetry module S(z_mod)
      - Informational coordinate psi (scalar)
    """

    def __init__(self,
                 input_channels: int = 1,
                 latent_dim: int = 64):
        super().__init__()
        self.encoder = Encoder(input_channels=input_channels,
                               latent_dim=latent_dim)
        self.attention = StructuralAttention(latent_dim=latent_dim)
        self.world_model = WorldModel(latent_dim=latent_dim,
                                      output_channels=input_channels)
        self.symmetry = SymmetryModule(latent_dim=latent_dim)
        # psi is stored as a 0-dim tensor
        self.register_buffer("psi", torch.tensor(0.0))

    def forward(self, x: torch.Tensor):
        # Encode
        z = self.encoder(x)
        # Structural attention with psi
        z_mod, gate = self.attention(z, self.psi)
        # World-model reconstruction
        x_pred = self.world_model(z_mod)
        # Reconstruction loss (for reporting)
        loss_recon = F.mse_loss(x_pred, x, reduction="mean")
        # Symmetry transform (not used in loss here, but available)
        z_sym = self.symmetry(z_mod)
        return x_pred, z, z_mod, z_sym, gate, loss_recon

    def step_psi(self, loss_recon: torch.Tensor,
                 baseline: float = 0.05,
                 alpha: float = 0.01):
        """Update psi using reconstruction loss."""
        self.psi = update_psi(self.psi, loss_recon,
                              baseline=baseline, alpha=alpha)
