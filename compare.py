import matplotlib.pyplot as plt
from PIL import Image

base = "data/mvtec_ad/bottle"
good = Image.open(f"{base}/train/good/000.png")
defect = Image.open(f"{base}/test/broken_large/000.png")
mask = Image.open(f"{base}/ground_truth/broken_large/000_mask.png")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(good)
axes[0].set_title("Good")
axes[1].imshow(defect)
axes[1].set_title("Defect: broken_large")
axes[2].imshow(mask, cmap="gray")
axes[2].set_title("Mask (white = defect)")
for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.savefig("compare.png")
print("Saved: compare.png")
