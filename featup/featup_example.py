import torch
import torchvision.transforms as T
from PIL import Image

from featup.util import norm, unnorm
from featup.plotting import plot_feats, plot_lang_heatmaps

def uae(f_map):
    print(f_map[0].shape)
    return torch.mean(f_map[0])

input_size = 224
# image_path = "sample-images/plant.png"
image_path = "/media/sarvagya-pc/2TB HDD/Balgrist/THS/images_combined_wp12s/BCN/S01/M01/axial/img_057.png"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
use_norm = True

transform = T.Compose([
    T.Resize(input_size),
    T.CenterCrop((input_size, input_size)),
    T.ToTensor(),
    norm
])

image_tensor = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
print(image_tensor.shape)

upsampler = torch.hub.load("mhamilton723/FeatUp", 'dino16', use_norm=use_norm).to(device)
hr_feats = upsampler(image_tensor)
lr_feats = upsampler.model(image_tensor)
plot_feats(unnorm(image_tensor)[0], lr_feats[0], hr_feats[0])

# print(upsampler)
print(hr_feats[0].shape)
print(lr_feats[0].shape)
# print(lr_feats[0,0])
# print(torch.mean(lr_feats[0,1]))