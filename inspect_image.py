import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image

from build_memory_bank import bank_path, extract_patch_features, make_preprocess


def load_memory_bank(category: str, size: int) -> torch.Tensor:
    path = bank_path(category, size)
    if not path.exists():
        raise SystemExit(f"{path} not found - run: "
                         f"python build_memory_bank.py --category {category} --size {size}")
    return torch.load(path)


def anomaly_map(image_path: Path, memory_bank: torch.Tensor, size: int) -> tuple[torch.Tensor, float]:
    """Return a size x size heatmap and the anomaly score of one image."""
    image = Image.open(image_path).convert("RGB")
    batch = make_preprocess(size)(image).unsqueeze(0)
    feats = extract_patch_features(batch)                # [1, h, w, 384]
    h, w = feats.shape[1], feats.shape[2]                # e.g. 28 x 28 for 224 px
    patches = feats.reshape(-1, feats.shape[-1])         # [h*w, 384]

    # For every patch: distance to its MOST SIMILAR good patch in the memory bank
    min_dist = torch.full((patches.shape[0],), float("inf"))
    for chunk in memory_bank.split(20000):          # in chunks to save RAM
        d = torch.cdist(patches, chunk)             # [h*w, chunk_size]
        min_dist = torch.minimum(min_dist, d.min(dim=1).values)

    heat = min_dist.reshape(1, 1, h, w)
    heat = F.interpolate(heat, size=(size, size), mode="bilinear", align_corners=False)[0, 0]
    return heat, min_dist.max().item()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one image and save a heatmap")
    parser.add_argument("image", type=Path, help="path to the image to inspect")
    parser.add_argument("--category", default="bottle", help="which memory bank to use")
    parser.add_argument("--size", type=int, default=224, help="image size used for the memory bank")
    parser.add_argument("--out", type=Path, default=Path("heatmap.png"))
    args = parser.parse_args()

    memory_bank = load_memory_bank(args.category, args.size)
    heat, score = anomaly_map(args.image, memory_bank, args.size)
    print(f"{args.image}  ->  anomaly score: {score:.2f}")

    image = Image.open(args.image).convert("RGB").resize((args.size, args.size))
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