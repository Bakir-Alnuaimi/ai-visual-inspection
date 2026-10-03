import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image

from build_memory_bank import bank_path, extract_patch_features, preprocess


def load_memory_bank(category: str) -> torch.Tensor:
    path = bank_path(category)
    if not path.exists():
        raise SystemExit(f"{path} not found - run: python build_memory_bank.py --category {category}")
    return torch.load(path)


def anomaly_map(image_path: Path, memory_bank: torch.Tensor) -> tuple[torch.Tensor, float]:
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
    parser = argparse.ArgumentParser(description="Inspect one image and save a heatmap")
    parser.add_argument("image", type=Path, help="path to the image to inspect")
    parser.add_argument("--category", default="bottle", help="which memory bank to use")
    parser.add_argument("--out", type=Path, default=Path("heatmap.png"))
    args = parser.parse_args()

    memory_bank = load_memory_bank(args.category)
    heat, score = anomaly_map(args.image, memory_bank)
    print(f"{args.image}  ->  anomaly score: {score:.2f}")

    image = Image.open(args.image).convert("RGB").resize((224, 224))
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
    plt.savefig(args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()