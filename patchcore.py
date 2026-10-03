"""
PatchCore-style anomaly detection - the whole "brain" in one file.

The idea (like a new worker in a factory):
  1. Look at many GOOD images and remember every small patch   -> memory bank
  2. For a new image, compare every patch with the memory bank:
     patch looks familiar = normal, patch looks new = DEFECT
"""
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.feature_extraction import create_feature_extractor

DATA_DIR = Path("data/mvtec_ad")    # the MVTec AD dataset
BANK_DIR = Path("memory_banks")     # one saved memory bank per category
IMAGE_SIZE = 224                    # every image is resized to 224 x 224

# "The eyes": a ResNet18 pretrained on ImageNet. We only use its middle layers.
_model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
_eyes = create_feature_extractor(_model, return_nodes=["layer2", "layer3"])

# Resize the WHOLE image (no cropping) and scale colors the way ResNet18 expects
_prepare = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_image(path) -> torch.Tensor:
    """Open an image file -> tensor [3, 224, 224]."""
    return _prepare(Image.open(path).convert("RGB"))


def patch_features(images: torch.Tensor) -> torch.Tensor:
    """Describe every patch of every image with 384 numbers -> [N, 28, 28, 384]."""
    with torch.no_grad():
        layers = _eyes(images)
    small = layers["layer2"]                                            # fine details   [N, 128, 28, 28]
    large = F.interpolate(layers["layer3"], size=small.shape[-2:],
                          mode="bilinear", align_corners=False)          # bigger context [N, 256, 28, 28]
    features = torch.cat([small, large], dim=1)                         # together       [N, 384, 28, 28]
    features = F.avg_pool2d(features, 3, stride=1, padding=1)           # each patch also looks at its neighbours
    return features.permute(0, 2, 3, 1)                                 # [N, 28, 28, 384]


def build_memory_bank(category: str) -> torch.Tensor:
    """Step 1 (learning): remember every patch of every GOOD training image."""
    paths = sorted((DATA_DIR / category / "train" / "good").glob("*.png"))
    if not paths:
        raise SystemExit(f"No training images for '{category}' - is it in {DATA_DIR}?")

    bank = []
    for i in range(0, len(paths), 16):                                  # 16 images at a time (saves RAM)
        images = torch.stack([load_image(p) for p in paths[i:i + 16]])
        features = patch_features(images)
        bank.append(features.reshape(-1, features.shape[-1]))           # grid of patches -> list of patches
        print(f"  {min(i + 16, len(paths))}/{len(paths)} images")
    return torch.cat(bank)


def bank_file(category: str) -> Path:
    return BANK_DIR / f"{category}.pt"


def save_memory_bank(bank: torch.Tensor, category: str) -> None:
    BANK_DIR.mkdir(exist_ok=True)
    torch.save(bank, bank_file(category))


def load_memory_bank(category: str) -> torch.Tensor:
    if not bank_file(category).exists():
        raise SystemExit(f"No memory bank for '{category}' - run: python train.py --category {category}")
    return torch.load(bank_file(category))


def detect(image_path, bank: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Step 2 (inspection): returns (heatmap [224, 224], anomaly score).
    The higher the score, the more likely the part is defective."""
    features = patch_features(load_image(image_path).unsqueeze(0))[0]   # [28, 28, 384]
    rows, cols, _ = features.shape
    patches = features.reshape(rows * cols, -1)                         # [784, 384]

    # For every patch: distance to the MOST SIMILAR good patch (bank is split into chunks to save RAM)
    closest = [torch.cdist(patches, chunk).min(dim=1).values for chunk in bank.split(20000)]
    distance = torch.stack(closest).min(dim=0).values                   # [784]

    heatmap = F.interpolate(distance.reshape(1, 1, rows, cols), size=(IMAGE_SIZE, IMAGE_SIZE),
                            mode="bilinear", align_corners=False)[0, 0]
    score = distance.max().item()                                       # the worst patch decides
    return heatmap, score
