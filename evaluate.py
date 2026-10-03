"""Test the model on ALL test images of a category and compute the image AUROC.

Usage:  python evaluate.py --category bottle
"""
import argparse

from sklearn.metrics import roc_auc_score

import patchcore

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--category", default="bottle")
args = parser.parse_args()

bank = patchcore.load_memory_bank(args.category)
truth, scores = [], []          # truth: 0 = good, 1 = defect   |   scores: what the model says

for folder in sorted((patchcore.DATA_DIR / args.category / "test").iterdir()):
    folder_scores = [patchcore.detect(path, bank)[1] for path in sorted(folder.glob("*.png"))]
    truth += [0 if folder.name == "good" else 1] * len(folder_scores)
    scores += folder_scores
    print(f"{folder.name:<20} {len(folder_scores):>3} images   "
          f"score min {min(folder_scores):.2f}  avg {sum(folder_scores) / len(folder_scores):.2f}  "
          f"max {max(folder_scores):.2f}")

print(f"\nImage AUROC ({args.category}): {roc_auc_score(truth, scores) * 100:.1f} %")
