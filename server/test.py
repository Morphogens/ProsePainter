import os
from typing import *

import numpy as np
import torchvision
from PIL import Image

from server.server_modelling import MaskOptimizer, ESRGAN
from server.server_modelling_utils import (
    process_mask,
    get_limits_from_mask,
    get_crop_tensor_from_img,
    scale_crop_tensor,
    merge_gen_img_into_canvas,
)

if __name__ == "__main__":
    lr = 0.5
    padding_percent = 10
    num_rec_steps = 16
    num_generations = 16

    esrgan = ESRGAN()
    num_chunks = 1

    mask_list = [
        Image.open("server/imgs/mask1.png"),
        Image.open("server/imgs/mask2.png"),
        Image.open("server/imgs/mask3.png"),
    ]

    style_prompt = "japanese painting"
    prompt_list = [
        "Roses in the sea",
        "The Great Wave off Kanagawa",
        "A small village near the sea",
    ]
    prompt_list = [prompt + " " + style_prompt for prompt in prompt_list]

    out_dir = "./test-generations"
    os.makedirs(out_dir, exist_ok=True)

    canvas_img_path = "server/imgs/canvas.jpeg"
    canvas_img = np.float32(
        np.asarray(Image.open(canvas_img_path).convert("RGB"))) / 255.

    img_height, img_width, _ch = canvas_img.shape
    target_img_size = (
        img_width,
        img_height,
    )

    for prompt, mask in zip(prompt_list, mask_list):
        mask.save(os.path.join(out_dir,
                               f"{'_'.join(prompt.split())}_mask.png"))

        mask = process_mask(
            mask,
            target_img_size,
        )

        Image.fromarray(np.uint8(mask * 255)).save(
            os.path.join(out_dir,
                         f"{'_'.join(prompt.split())}_processed_mask.jpg"))

        crop_limits = get_limits_from_mask(
            mask,
            padding_percent,
        )

        img_crop_tensor = get_crop_tensor_from_img(
            canvas_img,
            crop_limits,
        )
        img_crop_tensor = scale_crop_tensor(img_crop_tensor)

        mask_crop_tensor = get_crop_tensor_from_img(
            mask[..., None],
            crop_limits,
        )
        mask_crop_tensor = scale_crop_tensor(mask_crop_tensor)

        mask_optimizer = MaskOptimizer(
            prompt=prompt,
            cond_img=img_crop_tensor,
            mask=mask_crop_tensor,
            lr=lr,
            style_prompt=style_prompt,
        )

        mask_optimizer.optimize_reconstruction(num_iters=num_rec_steps, )

        rec_img = mask_optimizer.model.get_img_from_latents(
            mask_optimizer.gen_latents, )

        rec_img_pil = torchvision.transforms.ToPILImage(mode="RGB")(rec_img[0])
        rec_img_pil.save(
            os.path.join(out_dir,
                         f"rec_canvas_{'_'.join(prompt.split())}.jpg"))

        gen_img = None
        optim_step = 0
        for optim_step in range(num_generations):
            gen_img = mask_optimizer.optimize()

            updated_canvas = merge_gen_img_into_canvas(
                gen_img,
                mask_crop_tensor,
                canvas_img,
                crop_limits,
            )

            updated_canvas_pil = Image.fromarray(np.uint8(updated_canvas *
                                                          255))
            gen_img_pil = torchvision.transforms.ToPILImage(mode="RGB")(
                gen_img[0])
            gen_img_pil.save(
                os.path.join(out_dir,
                             f"{'_'.join(prompt.split())}_{optim_step}.jpg"))

            updated_canvas_pil.save(
                os.path.join(
                    out_dir,
                    f"canvas_{'_'.join(prompt.split())}_{optim_step}.jpg"))

            canvas_img = updated_canvas

        img_crop_tensor = get_crop_tensor_from_img(
            updated_canvas,
            crop_limits,
        )
        img_crop_tensor = scale_crop_tensor(img_crop_tensor)
        upscaled_crop = esrgan.upscale_img(
            img_crop_tensor,
            num_chunks,
        )

        updated_canvas = merge_gen_img_into_canvas(
            upscaled_crop,
            mask_crop_tensor,
            canvas_img,
            crop_limits,
        )

        updated_canvas_pil = Image.fromarray(np.uint8(updated_canvas * 255))
        updated_canvas_pil.save(
            os.path.join(
                out_dir,
                f"upscaled_canvas_{'_'.join(prompt.split())}_{optim_step}.jpg")
        )
