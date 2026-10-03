import argparse

from sklearn.metrics import roc_auc_score

from build_memory_bank import DATA_ROOT
from inspect_image import anomaly_map, load_memory_bank


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one MVTec AD category (image AUROC)")
    parser.add_argument("--category", default="bottle")
    args = parser.parse_args()

    memory_bank = load_memory_bank(args.category)
    test_dir = DATA_ROOT / args.category / "test"

    labels = []   # 0 = good, 1 = defect (the "truth")
    scores = []   # what our model says

    print(f"Category: {args.category}")
    for class_dir in sorted(test_dir.iterdir()):
        class_scores = []
        for image_path in sorted(class_dir.glob("*.png")):
            _, score = anomaly_map(image_path, memory_bank)
            class_scores.append(score)
            labels.append(0 if class_dir.name == "good" else 1)
            scores.append(score)

        print(f"{class_dir.name:<20} {len(class_scores):>3} images   "
              f"score min {min(class_scores):5.2f}   "
              f"avg {sum(class_scores) / len(class_scores):5.2f}   "
              f"max {max(class_scores):5.2f}")

    auroc = roc_auc_score(labels, scores)
    print(f"\nImage AUROC ({args.category}): {auroc * 100:.1f} %")


if __name__ == "__main__":
    main()