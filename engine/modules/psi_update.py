import torch


def update_psi(psi: torch.Tensor,
               loss_recon: torch.Tensor,
               baseline: float = 0.05,
               alpha: float = 0.01) -> torch.Tensor:
    """Update rule for the informational coordinate psi.

    psi_{t+1} = psi_t + alpha * (loss_recon - baseline)
    """
    if psi.dim() == 0:
        psi_val = psi
    else:
        psi_val = psi.mean()
    delta = alpha * (loss_recon.detach() - baseline)
    return psi_val + delta
