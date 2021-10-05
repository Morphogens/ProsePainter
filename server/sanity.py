import sys

sys.path.append("HuggingGAN")

import torchvision
import numpy as np
from PIL import Image

from server.vqgan import LayerLoss, LayeredGenerator
from server.webserver import Layer

num_optimizations = 100
lr = 0.5
target_img_size = 256

mask1 = np.array(Image.open("server/mask1.png"))
mask2 = np.array(Image.open("server/mask2.png"))

layer_list = []
for mask in [mask1, mask2]:
    layer = Layer(
        color=000,
        strength=1,
        prompt="a pink cute dog",
        img=mask,
    )
    layer_list.append(layer)

    break

layered_generator = LayeredGenerator(
    layer_list,
    target_img_size=target_img_size,
    lr=lr,
)

for step in range(num_optimizations):
    rec_img, loss_dict, _state = layered_generator.optimize()

    for key, value in loss_dict.items():
        print(key, "loss -->", value)

    rec_img_pil = torchvision.transforms.ToPILImage(mode='RGB')(rec_img[0])
    rec_img_pil.save(f"{step}.jpg")
