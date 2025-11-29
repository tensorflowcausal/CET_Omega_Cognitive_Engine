import torch
import torch.nn as nn


class WorldModel(nn.Module):
    """Simple decoder/world-model.

    Maps latent vectors back to image space: (B, latent_dim) -> (B, C, H, W).
    Default output is 1x28x28 (e.g. MNIST-like).
    """

    def __init__(self, latent_dim: int = 64, output_channels: int = 1):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_channels = output_channels
        self.fc = nn.Linear(latent_dim, 64 * 3 * 3)

        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2)
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2)
        self.deconv3 = nn.ConvTranspose2d(16, output_channels, kernel_size=4, stride=2)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(-1, 64, 3, 3)
        x = torch.relu(self.deconv1(x))
        x = torch.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))
        # crop or pad to 28x28 if needed
        if x.size(-1) > 28:
            x = x[..., :28, :28]
        return x
