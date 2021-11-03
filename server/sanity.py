import sys
from typing import *

import torch
import numpy as np
from PIL import Image

from server.server_model_utils import LayerOptimizer
from server.server_data_utils import (
    process_mask,
    get_limits_from_mask,
    get_crop_from_limits,
    scale_crop,
    merge_gen_img_into_canvas,
)

if __name__ == "__main__":
    lr = 0.5
    iters_per_mask = 4

    style_prompt = "japanese painting 4K"

    prompt_list = [
        "Waves",
        "Roses",
        "Plants",
        "Water drops",
    ]

    prompt_list = [prompt + " " + style_prompt for prompt in prompt_list]

    canvas_img = np.float32(
        np.asarray(Image.open("server/imgs/img.png").convert("RGB"))) / 255.

    img_height, img_width, _ch = canvas_img.shape
    target_img_size = (
        img_width,
        img_height,
    )

    mask_list = [
        Image.open("server/imgs/mask-waves.png"),
        Image.open("server/imgs/mask-roses.png"),
        Image.open("server/imgs/mask-plants.png"),
        Image.open("server/imgs/mask-drops.png"),
    ]

    mask_list = [process_mask(
        mask,
        target_img_size,
    ) for mask in mask_list]

    for prompt, mask in zip(prompt_list, mask_list):
        crop_limits = get_limits_from_mask(mask, )

        img_crop = get_crop_from_limits(
            canvas_img,
            crop_limits,
        )
        img_crop = scale_crop(img_crop)

        mask_crop = get_crop_from_limits(
            mask[..., None],
            crop_limits,
        )
        mask_crop = scale_crop(mask_crop)

        layer = LayerOptimizer(
            prompt=prompt,
            cond_img=img_crop,
            mask=mask_crop,
            lr=lr,
        )

        for step in range(iters_per_mask, ):
            gen_img = layer.optimize()

            # gen_img_pil = torchvision.transforms.ToPILImage(mode="RGB")(
            #     gen_img[0])
            # gen_img_pil.save(
            #     f"generations/{'_'.join(prompt.split())}_{step}.jpg")

            updated_canvas = merge_gen_img_into_canvas(
                gen_img,
                mask_crop,
                canvas_img,
                crop_limits,
            )

            Image.fromarray(np.uint8(updated_canvas * 255)).save(
                f"generations/final_{'_'.join(prompt.split())}_{step}.jpg")

    Image.fromarray(np.uint8(canvas_img * 255)).save(f"generations/final.jpg")
