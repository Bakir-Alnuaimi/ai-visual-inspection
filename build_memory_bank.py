from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.feature_extraction import create_feature_extractor

DATA_DIR = Path("data/mvtec_ad/bottle/train/good")
OUT_FILE = Path("memory_bank.pt")
IMAGE_SIZE = 224
BATCH_SIZE = 16

# 1) Pretrained network, cut after layer2 and layer3
model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
extractor = create_feature_extractor(model, return_nodes=["layer2", "layer3"])

# 2) Image preparation: resize the WHOLE image (no cropping) + ImageNet normalization
preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def extract_patch_features(batch: torch.Tensor) -> torch.Tensor:
    """Return one 384-number fingerprint per patch: shape [N, 28, 28, 384]."""
    with torch.no_grad():
        feats = extractor(batch)
    f2 = feats["layer2"]                                   # [N, 128, 28, 28]
    f3 = feats["layer3"]                                   # [N, 256, 14, 14]
    f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
    f = torch.cat([f2, f3], dim=1)                         # [N, 384, 28, 28]
    f = F.avg_pool2d(f, kernel_size=3, stride=1, padding=1)  # look at neighbours too
    return f.permute(0, 2, 3, 1)                           # [N, 28, 28, 384]


def main() -> None:
    paths = sorted(DATA_DIR.glob("*.png"))
    print(f"Found {len(paths)} good training images")

    all_patches = []
    for i in range(0, len(paths), BATCH_SIZE):
        chunk = paths[i:i + BATCH_SIZE]
        batch = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in chunk])
        feats = extract_patch_features(batch)
        all_patches.append(feats.reshape(-1, feats.shape[-1]))  # flatten to [N*784, 384]
        print(f"  processed {i + len(chunk):>3}/{len(paths)} images")

    memory_bank = torch.cat(all_patches)
    print(f"Memory bank shape: {tuple(memory_bank.shape)}")
    print(f"Memory bank size : {memory_bank.numel() * 4 / 1024**2:.0f} MB")

    torch.save(memory_bank, OUT_FILE)
    print(f"Saved: {OUT_FILE}")


if __name__ == "__main__":
    main()