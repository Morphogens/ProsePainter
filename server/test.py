import copy
import os
import glob
import subprocess
import itertools
from typing import *

import torch
import torchvision
import numpy as np
from loguru import logger
from PIL import Image

from server.server_modelling import MaskOptimizer
from server.server_modelling_utils import (
    process_mask,
    get_limits_from_mask,
    get_crop_tensor_from_img,
    scale_crop_tensor,
    merge_gen_img_into_canvas,
)
from server.server_config import MODEL_NAME


def get_clip_model_name_list_combination(
    clip_model_name_list,
    max_combination_num=None,
):
    if max_combination_num is None:
        max_combination_num = len(clip_model_name_list)

    clip_model_name_lists = []
    for binary_mask in itertools.product(range(2),
                                         repeat=len(clip_model_name_list)):
        if 1 not in binary_mask or sum(binary_mask) > max_combination_num:
            continue

        clip_model_name_lists.append(
            list(itertools.compress(clip_model_name_list, binary_mask)))

    return clip_model_name_lists


clip_model_name_lists = get_clip_model_name_list_combination(
    [
        "ViT-B/32",
        "ViT-B/16",
        "RN50x16",
        "RN50x4",
    ],
    max_combination_num=2,
)


def optimize(
    canvas_img,
    mask,
    target_img_size,
    clip_model_name_list,
    num_generations,
    lr,
    num_rec_steps,
    padding_percent,
    style_prompt,
    step,
    out_dir,
):
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
        model_name=MODEL_NAME,
        model_params_dict={
            'clip_model_name_list': clip_model_name_list,
        },
        recompute_model=True,
    )

    mask_optimizer.optimize_reconstruction(num_iters=num_rec_steps, )

    # rec_img = mask_optimizer.model.get_img_from_latents(
    #     mask_optimizer.gen_latents, )

    # rec_img_pil = torchvision.transforms.ToPILImage(mode="RGB")(
    #     rec_img[0])
    # rec_img_pil.save(
    #     os.path.join(
    #         out_dir,
    #         f"{counter:03d}_rec_canvas_{'_'.join(prompt.split())}.jpg"))

    gen_img = None
    for optim_step in range(num_generations):
        gen_img = mask_optimizer.optimize()

        updated_canvas = merge_gen_img_into_canvas(
            gen_img,
            mask_crop_tensor,
            canvas_img,
            crop_limits,
        )

        updated_canvas_pil = Image.fromarray(np.uint8(updated_canvas * 255))
        # gen_img_pil = torchvision.transforms.ToPILImage(mode="RGB")(gen_img[0])
        # gen_img_pil.save(
        #     os.path.join(
        #         out_dir,
        #         f"{'_'.join(prompt.split())}_{optim_step:03d}.jpg"))

        updated_canvas_pil.save(
            os.path.join(
                out_dir,
                f"{step:04d}_{'_'.join(prompt.split())}_{optim_step:03d}.jpg"))

    return updated_canvas


if __name__ == "__main__":
    num_rec_steps = 8
    num_generations = 16
    lr_list = [0.1, 0.25, 0.5]
    padding_percent_list = [10, 30]
    style_prompt_list = ["ArtStation HD", "Unreal Engine"]

    test_name = "frog"
    img_dir = os.path.join("./server/test-imgs", test_name)

    out_dir = "./test-generations"
    os.makedirs(out_dir, exist_ok=True)

    mask_path_list = glob.glob(os.path.join(img_dir, "mask-*"))
    mask_info_list = [
        tuple(mask_path.split("/")[-1].split("-"))
        for mask_path in mask_path_list
    ]
    mask_info_list.sort(key=lambda x: x[1])
    prompt_list = [
        " ".join(mask_info[2].split('.')[0].split("_"))
        for mask_info in mask_info_list
    ]
    mask_list = [
        Image.open(os.path.join(img_dir, '-'.join(mask_info)))
        for mask_info in mask_info_list
    ]

    canvas_img_path = glob.glob(os.path.join(img_dir, "canvas*"))[0]
    canvas_img = np.float32(
        np.asarray(Image.open(canvas_img_path).convert("RGB"))) / 255.

    original_canvas_img = copy.deepcopy(canvas_img)

    img_height, img_width, _ch = canvas_img.shape
    target_img_size = (
        img_width,
        img_height,
    )

    for clip_model_name_list in clip_model_name_lists:
        for style_prompt in style_prompt_list:
            for padding_percent in padding_percent_list:
                for lr in lr_list:
                    counter = 0
                    for prompt, mask in zip(prompt_list, mask_list):
                        prompt = f"{prompt} {style_prompt}"
                        try:
                            updated_canvas = optimize(
                                canvas_img=canvas_img,
                                mask=mask,
                                target_img_size=target_img_size,
                                clip_model_name_list=clip_model_name_list,
                                lr=lr,
                                num_generations=num_generations,
                                num_rec_steps=num_rec_steps,
                                padding_percent=padding_percent,
                                style_prompt=style_prompt,
                                step=counter,
                                out_dir=out_dir,
                            )
                            canvas_img = updated_canvas

                        except Exception as e:
                            logger.error(
                                "FAILEDDD!" +
                                f" {num_generations}_generations_{num_rec_steps}_rec_pad_{padding_percent}_using-{'-'.join([s.replace('/', '') for s in clip_model_name_list])}_lr_{lr}"
                                + repr(e), )
                            break

                        counter += 1

                    if counter == 0:
                        break

                    else:
                        try:
                            canvas_img = copy.deepcopy(original_canvas_img)

                            fps = 8

                            clip_models_str = '-'.join([
                                s.replace('/', '')
                                for s in clip_model_name_list
                            ])
                            lr_str = str(lr).replace('.', ',')
                            style_prompt_str = '-'.join(
                                style_prompt.split(' '))

                            out_filename = (f"using_{clip_models_str}"
                                            f"_style_{style_prompt_str}"
                                            f"_lr_{lr_str}"
                                            f"_pad_{padding_percent}"
                                            f"_{num_rec_steps}_rec_steps"
                                            f"{num_generations}_generations")

                            cmd = ("ffmpeg -y "
                                   "-r 16 "
                                   f"-pattern_type glob -i '{out_dir}/0*.jpg' "
                                   "-vcodec libx264 "
                                   f"-crf {fps} "
                                   "-pix_fmt yuv420p "
                                   "-vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' "
                                   f"{out_dir}/{out_filename}.mp4; "
                                   f"rm -r {out_dir}/0*.jpg;")

                            subprocess.check_call(cmd, shell=True)

                            updated_canvas_pil = Image.fromarray(
                                np.uint8(updated_canvas * 255))
                            updated_canvas_pil.save(
                                f"{out_dir}/{out_filename}.jpg")

                        except Exception as e:
                            logger.error("SAVING FAILED")
                            logger.error(repr(e))
