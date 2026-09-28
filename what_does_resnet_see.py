import torch
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights

# 1) Load the pretrained network
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.eval()

# 2) Prepare the image exactly like during ImageNet training
preprocess = weights.transforms()
image = Image.open("data/mvtec_ad/bottle/train/good/000.png").convert("RGB")
batch = preprocess(image).unsqueeze(0)
print("Input shape:", batch.shape)

# 3) Ask the network
with torch.no_grad():
    probs = model(batch).softmax(dim=1)[0]

# 4) Show the top 5 guesses
top5 = probs.topk(5)
categories = weights.meta["categories"]
for p, idx in zip(top5.values, top5.indices):
    print(f"{categories[idx]:<20} {p.item() * 100:5.1f} %")
