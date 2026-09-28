import torch
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.feature_extraction import create_feature_extractor

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights).eval()

# Cut the network: only return the outputs of layer2 and layer3
extractor = create_feature_extractor(model, return_nodes=["layer2", "layer3"])

image = Image.open("data/mvtec_ad/bottle/train/good/000.png").convert("RGB")
batch = weights.transforms()(image).unsqueeze(0)

with torch.no_grad():
    features = extractor(batch)

print("Input :", batch.shape)
for name, f in features.items():
    print(f"{name}: {f.shape}")
