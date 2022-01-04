import gc
import time
import time
from typing import *
from threading import Thread

import torch
from loguru import logger
import numpy as np
from PIL import Image

from server.server_data_utils import pil_to_base64
from server.server_modeling import ModelFactory
from server.server_modeling_utils import merge_gen_img_into_canvas

model_factory = ModelFactory()


class OptimizationManager:
    def __init__(
        self,
        batch_size=2,
        max_wait: float = 1,
    ):
        self.batch_size = batch_size
        self.max_wait = max_wait

        self.job_list = []
        self.num_iterations = 20
        self.lr = 0.15
        self.resolution = (400, 400)
        self.num_crops = 32
        self.num_accum_steps = 1

        self.num_crops = max(
            1,
            int(self.num_crops / self.num_accum_steps),
        )

        # NOTE: needs to be set from outside
        self.async_manager = None

        self.model_name = "taming"

        self.num_devices = torch.cuda.device_count()

        self.generator_dict = {}
        for cuda_idx in range(self.num_devices):
            generator = model_factory.load_model(
                self.model_name,
                device=f"cuda:{cuda_idx}",
                model_params_dict={
                    "clip_model_name_list": [
                        "ViT-B/32",
                        #"ViT-B/16",
                    ]
                },
            )
            self.generator_dict[f"cuda:{cuda_idx}"] = generator.to(f"cuda:{cuda_idx}")

    def start(self, ):
        Thread(target=self.taming_worker, ).start()

    def add_job(
        self,
        user_id: str,
        prompt: str,
        cond_img: torch.Tensor,
        mask: torch.Tensor,
        mask_crop_tensor: torch.Tensor,
        canvas_img: torch.Tensor,
        crop_limits: torch.Tensor,
    ):
        user_params_dict = {
            'user_id': user_id,
            'prompt': prompt,
            'cond_img': cond_img,
            'mask': mask,
            'mask_crop_tensor': mask_crop_tensor,
            'canvas_img': canvas_img,
            'crop_limits': crop_limits,
        }

        self.async_manager.activate_user(user_id)

        self.job_list.append(user_params_dict, )

        return

    def remove_job(
        self,
        user_id: str,
    ):
        self.async_manager.deactivate_async_user(user_id, )

        return

    def taming_worker(self, ):
        current_num_jobs = len(self.job_list)
        job_thread_list = [None]*self.num_devices
        t = None
        while True:
            if current_num_jobs > 0 and t is None:
                t = time.time()

            # logger.info(f"{current_num_jobs} WAITING FOR DATA TO BATCH...")
            time.sleep(0.5)

            self.job_list = [
                job for job in self.job_list
                if self.async_manager.user_active_dict[job["user_id"]]
            ]

            if t is not None:
                time_waited = time.time() - t
            else:
                time_waited = -1

            # logger.info(f"TIME WAITED {time_waited}/{self.max_wait} SECONDS")

            current_num_jobs = len(self.job_list)
            # print("THREAD LIST", job_thread_list)
            if current_num_jobs >= self.batch_size or time_waited > self.max_wait:
                t = None

                for cuda_idx in range(self.num_devices):
                    if current_num_jobs == 0:
                        continue

                    current_thread = job_thread_list[cuda_idx]

                    print("CURRENT THREAD", current_thread)
                    if current_thread is not None:
                        if current_thread.is_alive():
                            print(f"WAITING FOR WORKER {cuda_idx} TO FINISH!!!")
                            continue


                    #try:
                    limit_job_idx = max(1, min(int(current_num_jobs / self.num_devices), self.batch_size))
                    jobs_to_optimize = self.job_list[0:limit_job_idx]
                    self.job_list = self.job_list[limit_job_idx::]
                    
                    logger.info(f"BATCHING!!! in machine {cuda_idx}")
                    logger.info(f"NUMBER OF JOBS OPTIMIZING", len(jobs_to_optimize))
                    logger.info(f"NUMBER OF JOBS LEFT", len(self.job_list))
                    
                    job_thread = Thread(target=self.batched_optimization, args=(jobs_to_optimize, f"cuda:{cuda_idx}",))
                    job_thread_list[cuda_idx] = job_thread
                    job_thread.start()

                    #except Exception as e:
                    #    logger.error("ERROR IN BATCHED OPTIMIZATION!!")
                    #    logger.error(repr(e))


                    current_num_jobs = len(self.job_list)

    def batched_optimization(
        self,
        optim_job_list,
        device: str = "cuda:0",
    ):
        generator = self.generator_dict[device]
        
        max_resolution = (0,0)
        for user_params in optim_job_list:
            cond_img = user_params["cond_img"]
            cond_img = cond_img.to(device)

            cond_img_h = cond_img.shape[2]
            cond_img_w = cond_img.shape[3]
            x_pad_percent = 1 / min(1, cond_img_w / cond_img_h)
            y_pad_percent = 1 / min(1, cond_img_h / cond_img_w)
            x_pad_size = cond_img_w * (x_pad_percent - 1)
            y_pad_size = cond_img_h * (y_pad_percent - 1)

            cond_img = torch.nn.functional.pad(
                cond_img,
                (
                    int(x_pad_size / 2),
                    int(x_pad_size / 2),
                    int(y_pad_size / 2),
                    int(y_pad_size / 2),
                ),
                mode='constant',
                value=torch.tensor(0).to(device),
            )

            current_resolution = cond_img.shape[2::]

            if max(current_resolution) > max(max_resolution):
                max_resolution = current_resolution

            if max(max_resolution) >= max(self.resolution):
                max_resolution = self.resolution
                break


        print("RESOLUTION", max_resolution)

        latents_list = []
        user_prompt_list = []
        for user_params in optim_job_list:
            prompt = user_params["prompt"]
            user_prompt_list.append(prompt)

            cond_img = user_params["cond_img"]
            cond_img = cond_img.to(device)

            cond_img_h = cond_img.shape[2]
            cond_img_w = cond_img.shape[3]
            x_pad_percent = 1 / min(1, cond_img_w / cond_img_h)
            y_pad_percent = 1 / min(1, cond_img_h / cond_img_w)
            x_pad_size = cond_img_w * (x_pad_percent - 1)
            y_pad_size = cond_img_h * (y_pad_percent - 1)

            cond_img = torch.nn.functional.pad(
                cond_img,
                (
                    int(x_pad_size / 2),
                    int(x_pad_size / 2),
                    int(y_pad_size / 2),
                    int(y_pad_size / 2),
                ),
                mode='constant',
                value=torch.tensor(0).to(device),
            )

            current_resolution = cond_img.shape[2::]


            cond_img = torch.nn.functional.interpolate(
                cond_img,
                max_resolution,
                mode="bilinear",
            )

            with torch.no_grad():
                latents = generator.get_latents_from_img(cond_img, )

            latents_list.append(latents)

        text_logits_list = generator.get_clip_text_encodings(
            user_prompt_list, )

        batched_latents = torch.cat(latents_list, ).to(device)
        batched_latents = batched_latents.detach()
        batched_latents = torch.nn.Parameter(batched_latents)

        optimizer = torch.optim.Adam(
            params=[batched_latents],
            lr=self.lr,
            betas=(0.9, 0.999),
        )

        step = 0
        for _iter_idx in range(self.num_iterations):
            for _accum_idx in range(self.num_accum_steps):
                gen_img = generator.get_img_from_latents(batched_latents, )

                img_aug = generator.augment(
                    gen_img,
                    num_crops=self.num_crops,
                )
                img_logits_list = generator.get_clip_img_encodings(img_aug, )

                loss = 0

                for img_logits, text_logits in zip(img_logits_list,
                                                   text_logits_list):
                    text_logits = text_logits.repeat(self.num_crops, 1)
                    text_logits = text_logits.detach()

                    clip_loss = -10 * torch.cosine_similarity(
                        text_logits, img_logits).mean()

                    logger.info(f"CLIP LOSS {clip_loss}")
                    loss += clip_loss

                    loss.backward(retain_graph=False, )

            optimizer.step()
            optimizer.zero_grad()

            step += 1

            optim_job_list = [
                job for job in optim_job_list
                if self.async_manager.user_active_dict[job["user_id"]]
            ]

            for user_idx, user_params in enumerate(optim_job_list):
                user_id = user_params["user_id"]
                mask_crop_tensor = user_params["mask_crop_tensor"]
                canvas_img = user_params["canvas_img"]
                crop_limits = user_params["crop_limits"]
                cond_img = user_params["cond_img"]

                cond_img_h = cond_img.shape[2]
                cond_img_w = cond_img.shape[3]
                pad_scale = max(self.resolution) / max(cond_img_h, cond_img_w)
                x_pad_percent = 1 / min(1, cond_img_w / cond_img_h)
                y_pad_percent = 1 / min(1, cond_img_h / cond_img_w)
                x_pad_size = pad_scale * cond_img_w * (x_pad_percent - 1)
                y_pad_size = pad_scale * cond_img_h * (y_pad_percent - 1)

                with torch.no_grad():
                    img_rec = generator.get_img_from_latents(
                        batched_latents[user_idx, :][None, :], )

                img_rec_h = img_rec.shape[2]
                img_rec_w = img_rec.shape[3]

                img_rec = img_rec[:, :,
                                  int(y_pad_size /
                                      2):img_rec_h - int(y_pad_size / 2),
                                  int(x_pad_size / 2):img_rec_w -
                                  int(x_pad_size / 2), ]

                updated_canvas = merge_gen_img_into_canvas(
                    img_rec.to("cuda:0"),
                    mask_crop_tensor,
                    canvas_img,
                    crop_limits,
                )

                updated_canvas_pil = Image.fromarray(
                    np.uint8(updated_canvas * 255))
                updated_canvas_uri = pil_to_base64(updated_canvas_pil)

                self.async_manager.set_async_value(
                    user_id,
                    {
                        "message": "gen-results",
                        "user_id": user_id,
                        "image": updated_canvas_uri,
                        "step": step,
                        "num_iterations": self.num_iterations,
                        # "step": f"{step}/{self.num_iterations}"
                    },
                )

            torch.cuda.empty_cache()
            gc.collect()

        del optimizer
        del batched_latents
        torch.cuda.empty_cache()
        gc.collect()
