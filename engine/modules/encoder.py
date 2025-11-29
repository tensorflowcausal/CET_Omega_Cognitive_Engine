import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Simple convolutional encoder for image inputs.

    Input:  (B, C, H, W)
    Output: (B, latent_dim)
    """

    def __init__(self, input_channels: int = 1, latent_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.latent_dim = latent_dim
        self.flatten_dim = 64 * 3 * 3  # assumes input 28x28

        self.fc = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x expected shape: (B, C, 28, 28)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)  # 28 -> 14 -> 7 -> 3 (approx)
        x = x.view(x.size(0), -1)
        z = torch.tanh(self.fc(x))
        return z
