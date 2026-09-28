import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image

from build_memory_bank import extract_patch_features, preprocess

memory_bank = torch.load("memory_bank.pt")


def anomaly_map(image_path: Path) -> tuple[torch.Tensor, float]:
    """Return a 224x224 heatmap and the anomaly score of one image."""
    image = Image.open(image_path).convert("RGB")
    feats = extract_patch_features(preprocess(image).unsqueeze(0))  # [1, 28, 28, 384]
    patches = feats.reshape(-1, feats.shape[-1])                    # [784, 384]

    # For every patch: distance to its MOST SIMILAR good patch in the memory bank
    min_dist = torch.full((patches.shape[0],), float("inf"))
    for chunk in memory_bank.split(20000):          # in chunks to save RAM
        d = torch.cdist(patches, chunk)             # [784, chunk_size]
        min_dist = torch.minimum(min_dist, d.min(dim=1).values)

    heat = min_dist.reshape(1, 1, 28, 28)
    heat = F.interpolate(heat, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
    return heat, min_dist.max().item()


def main() -> None:
    image_path = Path(sys.argv[1])
    heat, score = anomaly_map(image_path)
    print(f"{image_path}  ->  anomaly score: {score:.2f}")

    image = Image.open(image_path).convert("RGB").resize((224, 224))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[1].imshow(heat, cmap="jet")
    axes[1].set_title(f"Heatmap (score {score:.2f})")
    axes[2].imshow(image)
    axes[2].imshow(heat, cmap="jet", alpha=0.5)
    axes[2].set_title("Overlay")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("heatmap.png")
    print("Saved: heatmap.png")


if __name__ == "__main__":
    main()