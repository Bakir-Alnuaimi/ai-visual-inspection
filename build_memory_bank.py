import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.feature_extraction import create_feature_extractor

DATA_ROOT = Path("data/mvtec_ad")
BANK_DIR = Path("memory_banks")
BATCH_SIZE = 16

# 1) Pretrained network, cut after layer2 and layer3
model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
extractor = create_feature_extractor(model, return_nodes=["layer2", "layer3"])


def make_preprocess(size: int) -> transforms.Compose:
    """Resize the WHOLE image (no cropping) to size x size + ImageNet normalization."""
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def extract_patch_features(batch: torch.Tensor) -> torch.Tensor:
    """Return one 384-number fingerprint per patch: shape [N, H/8, W/8, 384]."""
    with torch.no_grad():
        feats = extractor(batch)
    f2 = feats["layer2"]                                   # [N, 128, H/8, W/8]
    f3 = feats["layer3"]                                   # [N, 256, H/16, W/16]
    f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
    f = torch.cat([f2, f3], dim=1)                         # [N, 384, H/8, W/8]
    f = F.avg_pool2d(f, kernel_size=3, stride=1, padding=1)  # look at neighbours too
    return f.permute(0, 2, 3, 1)                           # [N, H/8, W/8, 384]


def bank_path(category: str, size: int) -> Path:
    """Where the memory bank is stored, e.g. memory_banks/bottle_224.pt"""
    return BANK_DIR / f"{category}_{size}.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the memory bank for one MVTec AD category")
    parser.add_argument("--category", default="bottle", help="e.g. bottle, screw, grid")
    parser.add_argument("--size", type=int, default=224, help="image size in pixels (multiple of 16)")
    args = parser.parse_args()
    preprocess = make_preprocess(args.size)

    data_dir = DATA_ROOT / args.category / "train" / "good"
    paths = sorted(data_dir.glob("*.png"))
    if not paths:
        raise SystemExit(f"No images found in {data_dir} - is the category downloaded?")
    print(f"[{args.category}, {args.size}px] Found {len(paths)} good training images")

    all_patches = []
    for i in range(0, len(paths), BATCH_SIZE):
        chunk = paths[i:i + BATCH_SIZE]
        batch = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in chunk])
        feats = extract_patch_features(batch)
        all_patches.append(feats.reshape(-1, feats.shape[-1]))  # flatten to [N*patches, 384]
        print(f"  processed {i + len(chunk):>3}/{len(paths)} images")

    memory_bank = torch.cat(all_patches)
    print(f"Memory bank shape: {tuple(memory_bank.shape)}")
    print(f"Memory bank size : {memory_bank.numel() * 4 / 1024**2:.0f} MB")

    out_file = bank_path(args.category, args.size)
    out_file.parent.mkdir(exist_ok=True)
    torch.save(memory_bank, out_file)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()