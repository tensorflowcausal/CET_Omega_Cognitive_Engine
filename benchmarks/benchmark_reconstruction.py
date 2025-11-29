"""Benchmark A: Reconstruction on MNIST.

Entrena el CET Ω Cognitive Engine como autoencoder sobre MNIST
y guarda:
- benchmarks/results/reconstruction_loss.png
- benchmarks/results/psi_evolution.png
"""

import os
import sys
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# AÑADIR LA RAÍZ DEL REPO AL PATH (soluciona ModuleNotFoundError: engine)
# ----------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from engine.cet_omega_cognitive_engine import CETOmegaCognitiveEngine
# (también funcionaría: from engine import CETOmegaCognitiveEngine)


def main():
    transform = T.Compose([T.ToTensor()])
    train = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
    loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CETOmegaCognitiveEngine(input_channels=1, latent_dim=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    psis = []

    for epoch in range(2):
        for x, _ in loader:
            x = x.to(device)

            x_pred, z, z_mod, z_sym, gate, loss_recon = model(x)

            optimizer.zero_grad()
            loss_recon.backward()
            optimizer.step()

            # actualizar psi
            model.step_psi(loss_recon)

            losses.append(loss_recon.item())
            psis.append(float(model.psi.item()))

        print(f"Epoch {epoch} | loss={losses[-1]:.4f} | psi={psis[-1]:.4f}")

    # Asegurar que existe la carpeta de resultados
    results_dir = os.path.join(REPO_ROOT, "benchmarks", "results")
    os.makedirs(results_dir, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Reconstruction loss")
    plt.title("CET Ω Engine: Reconstruction (MNIST)")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "reconstruction_loss.png"))

    plt.figure(figsize=(6, 4))
    plt.plot(psis)
    plt.xlabel("Iteration")
    plt.ylabel("psi")
    plt.title("CET Ω Engine: psi evolution")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "psi_evolution.png"))


if __name__ == "__main__":
    main()