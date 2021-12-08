import gc
import os
from typing import *

import torch
import torchvision
import numpy as np
from geniverse.models import TamingDecoder, Aphantasia
from loguru import logger
from upscaler.models import ESRGAN, ESRGANConfig

from server.server_config import MODEL_NAME, DEBUG, DEBUG_OUT_DIR, CLIP_MODEL_NAME_LIST

torch.manual_seed(123)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("DEVICE: ", DEVICE)


class ModelFactory:
    """
    Functionalities to load ready to use generative models.
    """
    def __init__(self, ) -> None:
        """
        Set up instances where models will be saved.
        """
        self.taming_decoder = None
        self.aphantasia = None
        self.esrgan = None

    def load_model(
        self,
        model_name: str,
        model_params_dict: Dict = None,
        recompute: bool = False,
    ) -> torch.nn.Module:
        """
        Load a model and store it to its respective class instance.

        Args:
            model_name (str): name of the model to load. Currently accepting `taming` and `aphantasia`.

        Returns:
            torch.nn.Module: ready to use model.
        """
        logger.debug(f"LOADING {model_name}...")

        if model_name == "taming":
            if self.taming_decoder is None or recompute:
                logger.info("SETTING UP TAMING...")
                self.taming_decoder = TamingDecoder(**model_params_dict, )
                self.taming_decoder.eval()

            model = self.taming_decoder

        elif model_name == "aphantasia":
            if self.aphantasia is None or recompute:
                logger.info("SETTING UP APHANTASIA...")
                self.aphantasia = Aphantasia(**model_params_dict, )
                self.aphantasia.eval()

            model = self.aphantasia

        if model_name == "esrgan":
            if self.esrgan is None or recompute:
                esrgan_config = ESRGANConfig()
                self.esrgan = ESRGAN(
                    esrgan_config,
                    **model_params_dict,
                )

            model = self.esrgan

        model = model.to(DEVICE)

        return model


model_factory = ModelFactory()


class MaskOptimizer:
    """
    Set parameters to optimize an masked image with text.
    """
    def __init__(
        self,
        prompt: str,
        cond_img: np.ndarray,
        mask: np.ndarray,
        lr: float,
        rec_lr: float = 0.1,
        style_prompt: str = "",
        model_name: str = None,
        model_params_dict: Dict = {},
        recompute_model: bool = False,
        **kwargs,
    ) -> None:
        """
        Set up optimization parameters.

        Args:
            prompt (str): prompt used to guide the optimization.
            cond_img (np.ndarray): initial image used to start the optimization.
            mask (np.ndarray): mask determining the region to optimize.
            lr (float): learning rate.
            rel_lr (float): learning rate for the reconstruction optimization.
            style_prompt (str): prompt representing the style to use for the generated image.
        """
        self.mask = mask.to(DEVICE)
        self.cond_img = cond_img.to(DEVICE)

        self.layer_size = mask.shape[2::]

        if model_name is None:
            model_name = MODEL_NAME

        self.model = model_factory.load_model(
            model_name,
            recompute=recompute_model,
            model_params_dict=model_params_dict,
        )

        # text_latents_list = self.model.get_clip_text_encodings(prompt, )
        # text_latents_list = [
        #     text_latents.detach().to(DEVICE)
        #     for text_latents in text_latents_list
        # ]
        # self.text_latents_list = text_latents_list

        logger.debug(f"STYLE PROMPT {style_prompt}")
        self.prompt = prompt
        self.style_prompt = style_prompt

        # self.style_latents_list = None
        # if style_prompt != "":
        #     style_latents_list = self.model.get_clip_text_encodings(prompt, )
        #     style_latents_list = [
        #         style_latents.detach().to(DEVICE)
        #         for style_latents in style_latents_list
        #     ]
        #     self.style_latents_list = style_latents_list

        self.gen_latents = self.model.get_latents_from_img(cond_img, )
        self.gen_latents = self.gen_latents.to(DEVICE)
        self.gen_latents = self.gen_latents.detach().clone()
        self.gen_latents.requires_grad = True
        self.gen_latents = torch.nn.Parameter(self.gen_latents)

        self.optimizer = torch.optim.AdamW(
            params=[self.gen_latents],
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=0.1,
        )

        self.rec_optimizer = torch.optim.AdamW(
            params=[self.gen_latents],
            lr=rec_lr,
            betas=(0.9, 0.999),
            weight_decay=0.1,
        )

        return

    def optimize_reconstruction(
        self,
        num_iters: int = 16,
    ) -> None:
        """
        Optimizes the current latent to maximize its similarity with the conditional image.

        Args:
            num_iters (int, optional): Number of optimization steps. Defaults to 16.
        """
        rec_mask = self.mask
        rec_mask[rec_mask > 0] = 1

        rec_loss_weight = self.cond_img.shape[2] * self.cond_img.shape[3]

        for _iter_idx in range(num_iters):
            gen_img = self.model.get_img_from_latents(self.gen_latents, )
            gen_img = (rec_mask * gen_img) + (1 - rec_mask) * self.cond_img

            loss = rec_loss_weight * torch.nn.functional.mse_loss(
                gen_img,
                self.cond_img,
            )
            logger.debug(f"MSE LOSS {loss}")

            self.rec_optimizer.zero_grad()
            loss.backward(retain_graph=False, )
            self.rec_optimizer.step()

        torch.cuda.empty_cache()
        gc.collect()

        return

    def optimize(self, ) -> torch.Tensor:
        """
        Perform one optimization step. Uses CLIP to maximize the encodings of the generated image and the text prompts provided.

        Returns:
            torch.Tensor: updated image.
        """
        loss = 0

        gen_img = self.model.get_img_from_latents(self.gen_latents, )
        gen_img = (self.mask * gen_img) + (1 - self.mask) * self.cond_img

        if DEBUG:
            os.makedirs(
                DEBUG_OUT_DIR,
                exist_ok=True,
            )

            torchvision.transforms.ToPILImage(mode="L")(self.mask[0]).save(
                os.path.join(DEBUG_OUT_DIR, "mask.jpg"))
            torchvision.transforms.ToPILImage(mode="RGB")(gen_img[0], ).save(
                os.path.join(DEBUG_OUT_DIR, "gen_img.jpg"))
            torchvision.transforms.ToPILImage(mode="RGB")(
                self.cond_img[0]).save(
                    os.path.join(DEBUG_OUT_DIR, "init_img.jpg"))

        img_aug = self.model.augment(
            gen_img,
            num_crops=32,
        )
        img_latents_list = self.model.get_clip_img_encodings(img_aug, )

        for latents_idx in range(len(img_latents_list)):
            text_latents = self.model.get_clip_text_encodings(
                self.prompt, )[latents_idx]
            img_latents = img_latents_list[latents_idx]
            loss += (text_latents - img_latents).norm(
                dim=-1).div(2).arcsin().pow(2).mul(2).mean()

            if self.style_prompt != "":
                style_latents = self.model.get_clip_text_encodings(
                    self.style_prompt, )[latents_idx]
                loss += (style_latents - img_latents).norm(
                    dim=-1).div(2).arcsin().pow(2).mul(2).mean()

            # TODO: integrate other losses
            # loss = -10 * torch.cosine_similarity(
            #     self.text_latents,
            #     img_latents,
            # ).mean()

            logger.debug(f"LOSS --> {loss} \n\n")

            def scale_grad(grad, ):
                grad_size = grad.shape[2:4]

                grad_mask = torch.nn.functional.interpolate(
                    self.mask,
                    grad_size,
                    mode="bilinear",
                )

                masked_grad = grad * grad_mask

                return masked_grad

            gen_img_hook = gen_img.register_hook(scale_grad, )

            self.optimizer.zero_grad()
            loss.backward(retain_graph=False, )
            self.optimizer.step()

        gen_img_hook.remove()

        gen_img = torch.nn.functional.interpolate(
            gen_img,
            self.layer_size,
            mode='bilinear',
            align_corners=True,
        )

        torch.cuda.empty_cache()
        gc.collect()

        return gen_img
