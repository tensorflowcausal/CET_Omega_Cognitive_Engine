"""Benchmark B: World-model prediction en MNIST.

Usa el CET Ω Cognitive Engine como autoencoder (world model simple)
y guarda una figura comparando imágenes originales vs reconstruidas:

- benchmarks/results/worldmodel_examples.png
"""

import os
import sys
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# AÑADIR LA RAÍZ DEL REPO AL PATH
# ----------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from engine.cet_omega_cognitive_engine import CETOmegaCognitiveEngine


def make_grid(x, x_pred, n=8):
    """Devuelve un grid de n imágenes originales y sus reconstrucciones."""
    x = x[:n].cpu().detach()
    x_pred = x_pred[:n].cpu().detach()

    rows = 2
    cols = n
    fig, axes = plt.subplots(rows, cols, figsize=(1.5 * cols, 3))

    for i in range(n):
        axes[0, i].imshow(x[i, 0], cmap="gray")
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("input")

        axes[1, i].imshow(x_pred[i, 0], cmap="gray")
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("recon")

    plt.tight_layout()
    return fig


def main():
    transform = T.Compose([T.ToTensor()])
    test = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )
    loader = torch.utils.data.DataLoader(test, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CETOmegaCognitiveEngine(input_channels=1, latent_dim=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Entrenamos MUY poquito solo para que las reconstrucciones no sean ruido
    for epoch in range(1):
        for x, _ in loader:
            x = x.to(device)
            x_pred, z, z_mod, z_sym, gate, loss_recon = model(x)

            optimizer.zero_grad()
            loss_recon.backward()
            optimizer.step()
        print(f"[worldmodel] Epoch {epoch} | loss={loss_recon.item():.4f}")

    # Tomamos un batch y generamos figura comparación input vs recon
    x, _ = next(iter(loader))
    x = x.to(device)
    with torch.no_grad():
        x_pred, *_ = model(x)

    results_dir = os.path.join(REPO_ROOT, "benchmarks", "results")
    os.makedirs(results_dir, exist_ok=True)

    fig = make_grid(x, x_pred, n=8)
    out_path = os.path.join(results_dir, "worldmodel_examples.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[worldmodel] Saved {out_path}")


if __name__ == "__main__":
    main()