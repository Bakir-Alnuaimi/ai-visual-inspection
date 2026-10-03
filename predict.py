"""Inspect ONE image: print its anomaly score and save a heatmap.

Usage:  python predict.py data/mvtec_ad/bottle/test/broken_large/000.png --category bottle
"""
import argparse

import matplotlib.pyplot as plt
from PIL import Image

import patchcore

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("image", help="path to the image to inspect")
parser.add_argument("--category", default="bottle", help="which memory bank to use")
parser.add_argument("--out", default="heatmap.png", help="where to save the result picture")
args = parser.parse_args()

bank = patchcore.load_memory_bank(args.category)
heatmap, score = patchcore.detect(args.image, bank)
print(f"Anomaly score: {score:.2f}")

# Left: original image - Right: image with the heatmap on top (red = suspicious)
image = Image.open(args.image).convert("RGB").resize((patchcore.IMAGE_SIZE, patchcore.IMAGE_SIZE))
fig, (left, right) = plt.subplots(1, 2, figsize=(8, 4))
left.imshow(image)
left.set_title("Image")
right.imshow(image)
right.imshow(heatmap, cmap="jet", alpha=0.5)
right.set_title(f"Heatmap (score {score:.2f})")
for ax in (left, right):
    ax.axis("off")
plt.tight_layout()
plt.savefig(args.out)
print(f"Saved: {args.out}")
