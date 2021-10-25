from typing import *

import torch
import numpy as np
from PIL import Image

from server.server_config import MAX_IMG_DIM


def process_mask(
    mask_pil: Image.Image,
    size: Tuple = None,
    min_thold: float = 0.1,
):
    mask_pil = mask_pil.convert("L")
    if size is not None:
        mask_pil = mask_pil.resize(size)

    mask = np.float32(np.array(mask_pil)) / 255.
    mask[mask < min_thold] = 0
    mask[mask > 1] = 1

    return mask


def get_limits_from_mask(
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


def get_crop_from_limits(
    img,
    limits,
):
    img_crop = img[limits[0]:limits[1], limits[2]:limits[3], ]
    img_crop = torch.tensor(img_crop)[None, ...].permute(0, 3, 1, 2)

    return img_crop


def scale_crop(crop, ):
    crop_size = crop.shape[2::]

    if any([size for size in crop_size]):
        scale_factor = max(crop_size) / MAX_IMG_DIM
        scale_factor = scale_factor
        crop_size = tuple(np.int32(np.asarray(crop_size) / scale_factor))

    crop = torch.nn.functional.interpolate(
        crop,
        crop_size,
        mode='bilinear',
        align_corners=True,
    )

    return crop


def merge_gen_img_into_canvas(
    gen_img,
    mask,
    canvas_img,
    crop_limits,
):
    if not torch.is_tensor(gen_img):
        gen_img = torch.tensor(gen_img[None, :].permute(0, 3, 1, 2))

    if not torch.is_tensor(mask):
        mask = torch.tensor(mask[None, None, ...])

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

    # torchvision.transforms.ToPILImage(mode="RGB")(
    #     gen_img[0]).save("generations/final_gen.png")
    # torchvision.transforms.ToPILImage(mode="L")(
    #     mask[0]).save("generations/final_mask.png")

    gen_img = gen_img[0].detach().cpu().permute(1, 2, 0).numpy()
    mask = mask[0].detach().cpu().permute(1, 2, 0).numpy()

    canvas_img[
        crop_limits[0]:crop_limits[1],
        crop_limits[2]:crop_limits[3], :] = canvas_img[
            crop_limits[0]:crop_limits[1],
            crop_limits[2]:crop_limits[3], :] * (1 - mask) + gen_img * mask

    return canvas_img
