from pathlib import Path

from sklearn.metrics import roc_auc_score

from inspect_image import anomaly_map

TEST_DIR = Path("data/mvtec_ad/bottle/test")


def main() -> None:
    labels = []   # 0 = good, 1 = defect (the "truth")
    scores = []   # what our model says

    for class_dir in sorted(TEST_DIR.iterdir()):
        class_scores = []
        for image_path in sorted(class_dir.glob("*.png")):
            _, score = anomaly_map(image_path)
            class_scores.append(score)
            labels.append(0 if class_dir.name == "good" else 1)
            scores.append(score)

        print(f"{class_dir.name:<15} {len(class_scores):>3} images   "
              f"score min {min(class_scores):5.2f}   "
              f"avg {sum(class_scores) / len(class_scores):5.2f}   "
              f"max {max(class_scores):5.2f}")

    auroc = roc_auc_score(labels, scores)
    print(f"\nImage AUROC: {auroc * 100:.1f} %")


if __name__ == "__main__":
    main()