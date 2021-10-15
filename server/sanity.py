import sys
from typing import *
from matplotlib.pyplot import sca
from torchvision.transforms.functional import scale

sys.path.append("HuggingGAN")

import torch
import torchvision
import numpy as np
from PIL import Image

from server.vqgan import Layer, LayeredGenerator

MAX_IMG_DIM = 512


def load_mask(
    mask_path: str,
    size: Tuple = None,
    min_thold: float = 0.1,
):
    mask_pil = Image.open(mask_path, ).convert("L")
    if size is not None:
        mask_pil = mask_pil.resize(size)

    mask = np.float32(np.array(mask_pil)) / 255.

    mask[mask < min_thold] = 0

    return mask


def get_crops_from_mask(
    mask: np.ndarray,
    padding_percent: int = 10,
):
    height, width = mask.shape
    w_pad = int(width * (padding_percent / 100))
    h_pad = int(height * (padding_percent / 100))

    w_accum = np.where(np.sum(
        mask,
        axis=0,
    ) > 0)[0]
    w_limits = (
        max(0, w_accum[0] - w_pad),
        min(width, w_accum[-1] + w_pad),
    )

    h_accum = np.where(np.sum(
        mask,
        axis=1,
    ) > 0)[0]
    h_limits = (
        max(0, h_accum[0] - h_pad),
        min(height, h_accum[-1] + h_pad),
    )

    return h_limits[0], h_limits[1], w_limits[0], w_limits[1]


def process_mask_img(
    mask,
    img,
    limits,
):
    mask = mask[limits[0]:limits[1], limits[2]:limits[3], ]
    img = img[limits[0]:limits[1], limits[2]:limits[3], ]

    img = torch.tensor(img)[None, ...].permute(0, 3, 1, 2)

    mask = torch.tensor(mask)[None, None, ...]
    mask[mask > 0] = 1

    return img, mask


def scale_img_mask(
    img,
    mask,
):
    img_size = img.shape[2::]
    if any([size for size in img_size]):
        scale_factor = max(img_size) / MAX_IMG_DIM
        scale_factor = scale_factor
        img_size = tuple(np.int32(np.asarray(img_size) / scale_factor))

    img = torch.nn.functional.interpolate(
        img,
        img_size,
        mode='bilinear',
        align_corners=True,
    )

    mask = torch.nn.functional.interpolate(
        mask,
        img_size,
        mode='bilinear',
        align_corners=True,
    )

    print(f"SCALE FACTOR {scale_factor}")

    return img, mask, scale_factor


if __name__ == "__main__":
    lr = 0.05

    num_optimizations = 1
    iters_per_mask = 50

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

    img_height, img_width, _c = canvas_img.shape
    target_img_size = (
        img_width,
        img_height,
    )

    mask_list = [
        load_mask("server/imgs/mask-waves.png", target_img_size),
        load_mask("server/imgs/mask-roses.png", target_img_size),
        load_mask("server/imgs/mask-plants.png", target_img_size),
        load_mask("server/imgs/mask-drops.png", target_img_size),
    ]

    crop_limits_list = [get_crops_from_mask(mask, ) for mask in mask_list]

    img_mask_list = [
        process_mask_img(
            mask,
            canvas_img,
            crop_limits,
        ) for mask, crop_limits in zip(mask_list, crop_limits_list)
    ]

    img_mask_scale_list = [
        scale_img_mask(
            img,
            mask,
        ) for img, mask in img_mask_list
    ]

    layer_list = []
    for prompt, img_mask_scale, crop_limits in zip(
            prompt_list,
            img_mask_scale_list,
            crop_limits_list,
    ):
        cond_img, mask, scale_factor = img_mask_scale
        layer = Layer(
            prompt=prompt,
            mask=mask,
            cond_img=cond_img,
            lr=lr,
        )
        for step in range(iters_per_mask, ):
            gen_img = layer.optimize()

            gen_img = torch.nn.functional.interpolate(
                gen_img,
                mask.shape[2::],
                mode='bilinear',
                align_corners=True,
            )

            gen_img_pil = torchvision.transforms.ToPILImage(mode="RGB")(
                gen_img[0])
            gen_img_pil.save(f"{'_'.join(prompt.split())}_{step}.jpg")

        gen_img = torch.nn.functional.interpolate(
            gen_img,
            (
                int((crop_limits[1] - crop_limits[0])),
                int((crop_limits[3] - crop_limits[2])),
            ),
            mode='bilinear',
            align_corners=True,
        )
        mask = torch.nn.functional.interpolate(
            mask,
            (
                int((crop_limits[1] - crop_limits[0])),
                int((crop_limits[3] - crop_limits[2])),
            ),
            mode='bilinear',
            align_corners=True,
        )

        gen_img = gen_img[0].detach().cpu().permute(1, 2, 0).numpy()
        mask_img = mask[0].permute(1, 2, 0).cpu().numpy()

        canvas_img[crop_limits[0]:crop_limits[1],
                   crop_limits[2]:crop_limits[3], :] = canvas_img[
                       crop_limits[0]:crop_limits[1],
                       crop_limits[2]:crop_limits[3], :] * (
                           1 - mask_img) + gen_img * mask_img
        Image.fromarray(np.uint8(
            canvas_img * 255)).save(f"final_{'_'.join(prompt.split())}.jpg")

    Image.fromarray(np.uint8(canvas_img * 255)).save(f"final.jpg")

    # layered_generator = LayeredGenerator(
    #     layer_list,
    #     target_img_size=target_img_size,
    #     lr=lr,
    #     cond_img=cond_img,
    # )

    # _ = layered_generator.optimize_scaled(
    #     scale_factor=2,
    #     num_iters=1000,
    # )

    # for optim_step in range(num_optimizations):
    #     img_rec, loss_dict, _state = layered_generator.optimize(
    #         iters_per_mask=iters_per_mask, )
