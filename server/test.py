import copy
import os
import gc
import subprocess
from typing import *

import torch
import torchvision
import numpy as np
from PIL import Image

from server.server_modelling import MaskOptimizer
from server.server_modelling_utils import (
    process_mask,
    get_limits_from_mask,
    get_crop_tensor_from_img,
    scale_crop_tensor,
    merge_gen_img_into_canvas,
)
from server.server_config import CLIP_MODEL_NAME_LIST

if __name__ == "__main__":
    lr_list = [0.1, 0.25, 0.5]
    padding_percent_list = [5, 10, 30]
    num_rec_steps = 8
    num_generations = 16

    # esrgan = ESRGAN()
    # num_chunks = 1

    mask_list = [
        Image.open("server/imgs/mask1.png"),
        Image.open("server/imgs/mask2.png"),
        Image.open("server/imgs/mask3.png"),
    ]

    style_prompt = "japanese painting"
    prompt_list = [
        "Roses in the sea",
        "The Great Wave of Kanagawa",
        "A small village near the sea",
    ]
    prompt_list = [prompt + " " + style_prompt for prompt in prompt_list]

    out_dir = "./test-generations"
    os.makedirs(out_dir, exist_ok=True)

    canvas_img_path = "server/imgs/canvas.jpeg"
    canvas_img = np.float32(
        np.asarray(Image.open(canvas_img_path).convert("RGB"))) / 255.

    original_canvas_img = copy.deepcopy(canvas_img)

    img_height, img_width, _ch = canvas_img.shape
    target_img_size = (
        img_width,
        img_height,
    )

    for lr in lr_list:
        for padding_percent in padding_percent_list:
            counter = 0
            for prompt, mask in zip(prompt_list, mask_list):
                # mask.save(
                #     os.path.join(out_dir, f"0_{'_'.join(prompt.split())}_mask.png"))

                mask = process_mask(
                    mask,
                    target_img_size,
                )

                # Image.fromarray(np.uint8(mask * 255)).save(
                #     os.path.join(
                #         out_dir,
                #         f"{counter:03d}_{'_'.join(prompt.split())}_processed_mask.jpg")
                # )

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

                mask_optimizer.optimize_reconstruction(
                    num_iters=num_rec_steps, )

                # rec_img = mask_optimizer.model.get_img_from_latents(
                #     mask_optimizer.gen_latents, )

                # rec_img_pil = torchvision.transforms.ToPILImage(mode="RGB")(
                #     rec_img[0])
                # rec_img_pil.save(
                #     os.path.join(
                #         out_dir,
                #         f"{counter:03d}_rec_canvas_{'_'.join(prompt.split())}.jpg"))

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

                    updated_canvas_pil = Image.fromarray(
                        np.uint8(updated_canvas * 255))
                    gen_img_pil = torchvision.transforms.ToPILImage(
                        mode="RGB")(gen_img[0])
                    # gen_img_pil.save(
                    #     os.path.join(
                    #         out_dir,
                    #         f"{'_'.join(prompt.split())}_{optim_step:03d}.jpg"))

                    updated_canvas_pil.save(
                        os.path.join(
                            out_dir,
                            f"{counter:04d}_canvas_{'_'.join(prompt.split())}_{optim_step:03d}.jpg"
                        ))

                    canvas_img = updated_canvas

                    counter += 1

                # img_crop_tensor = get_crop_tensor_from_img(
                #     updated_canvas,
                #     crop_limits,
                # )
                # img_crop_tensor = scale_crop_tensor(img_crop_tensor)
                # img_crop_tensor = esrgan.upscale_img(
                #     img_crop_tensor,
                #     num_chunks,
                # )

                # updated_canvas = merge_gen_img_into_canvas(
                #     img_crop_tensor,
                #     mask_crop_tensor,
                #     canvas_img,
                #     crop_limits,
                # )

                # updated_canvas_pil = Image.fromarray(np.uint8(updated_canvas * 255))
                # updated_canvas_pil.save(
                #     os.path.join(
                #         out_dir,
                #         f"upscaled_canvas_{'_'.join(prompt.split())}_{optim_step}.jpg")
                # )

            canvas_img = copy.deepcopy(original_canvas_img)

            fps = 8

            cmd = (
                "ffmpeg -y "
                "-r 16 "
                f"-pattern_type glob -i '{out_dir}/0*.jpg' "
                "-vcodec libx264 "
                f"-crf {fps} "
                "-pix_fmt yuv420p "
                f"{out_dir}/{num_generations}_generations_{num_rec_steps}_pad_{padding_percent}_rec_using-{'-'.join([s.replace('/', '') for s in CLIP_MODEL_NAME_LIST])}_lr_{lr}.mp4; "
                f"rm -r {out_dir}/0*.jpg;")

            subprocess.check_call(cmd, shell=True)

            updated_canvas_pil.save(
                os.path.join(
                    out_dir,
                    f"{num_generations}_generations_{num_rec_steps}_rec_pad_{padding_percent}_using-{'-'.join([s.replace('/', '') for s in CLIP_MODEL_NAME_LIST])}_lr_{lr}.jpg"
                ))
