from PIL import Image
from torchvision.transforms.functional import to_tensor

image = Image.open("data/mvtec_ad/bottle/train/good/000.png")
print("Size:", image.size, "Mode:", image.mode)

tensor = to_tensor(image)
print("Shape:", tensor.shape)
print("Pixel in the middle (R, G, B):", tensor[:, 450, 450])
