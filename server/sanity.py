import sys

sys.path.append("HuggingGAN")

import torchvision
import numpy as np
from PIL import Image

from server.vqgan import Layer, LayeredGenerator

iters_per_mask = 100
num_optimizations = 1
lr = 0.5
target_img_size = (128, 128)

# cond_img = Image.open('server/cond.png').resize(target_img_size)
cond_img = None

style_prompt = " 4k trending on artstation"

background_mask = np.ones(target_img_size + (4, ))
background_prompt = "Blue sea"

mask_list = [
    background_mask,
    # np.array(Image.open("server/mask1.png").resize(target_img_size)),
    # np.array(Image.open("server/mask2.png").resize(target_img_size)),
]

prompt_list = [
    background_prompt,
    # "Green grass",
    # "A cute dog",
]

prompt_list = [prompt + style_prompt for prompt in prompt_list]

cum_mask = np.zeros_like(background_mask)
for mask_idx, mask in enumerate(mask_list[::-1]):
    inv_mask_idx = len(mask_list) - mask_idx - 1
    if mask_idx > 0:
        mask_list[inv_mask_idx] = mask * (1 - cum_mask)

    cum_mask += mask
    cum_mask = cum_mask.clip(0, 1)

layer_list = []
for prompt, mask in zip(prompt_list, mask_list):
    layer = Layer(
        prompt=prompt,
        mask=mask,
        color=000,
        strength=1,
    )
    layer_list.append(layer)

layered_generator = LayeredGenerator(
    layer_list,
    target_img_size=target_img_size,
    lr=lr,
    cond_img=cond_img,
)

# _ = layered_generator.optimize_scaled(
#     scale_factor=2,
#     num_iters=1000,
# )

for optim_step in range(num_optimizations):
    img_rec, loss_dict, _state = layered_generator.optimize(
        iters_per_mask=iters_per_mask, )
