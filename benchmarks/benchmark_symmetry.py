"""Benchmark C: Sensibilidad a simetrías en el espacio latente.

Aplica rotaciones pequeñas a MNIST y mide cuánto cambia el embedding z.
Guarda un gráfico de dispersión:

- benchmarks/results/symmetry_sensitivity.png
"""

import os
import sys
import math
import torch
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


def main():
    base_transform = T.Compose([T.ToTensor()])
    dataset = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=base_transform,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CETOmegaCognitiveEngine(input_channels=1, latent_dim=32).to(device)
    model.eval()

    degrees = []
    deltas = []

    # Tomamos unos pocos batches para el experimento
    for i, (x, _) in enumerate(loader):
        if i >= 10:
            break

        x = x.to(device)

        for angle in [-15, -10, -5, 5, 10, 15]:
            # rotación
            theta = math.radians(angle)
            rot = T.functional.rotate(x.cpu(), angle=angle)
            rot = rot.to(device)

            with torch.no_grad():
                # obtenemos embeddings originales y rotados
                _, z, *_ = model(x)
                _, z_rot, *_ = model(rot)

            # distancia media en latente
            delta = torch.mean(torch.norm(z - z_rot, dim=1)).item()
            degrees.append(angle)
            deltas.append(delta)

    # Graficar
    results_dir = os.path.join(REPO_ROOT, "benchmarks", "results")
    os.makedirs(results_dir, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.scatter(degrees, deltas)
    plt.xlabel("Rotación [grados]")
    plt.ylabel("||z - z_rot|| promedio")
    plt.title("CET Ω Engine: sensibilidad a rotaciones en el espacio latente")
    plt.tight_layout()
    out_path = os.path.join(results_dir, "symmetry_sensitivity.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[symmetry] Saved {out_path}")


if __name__ == "__main__":
    main()