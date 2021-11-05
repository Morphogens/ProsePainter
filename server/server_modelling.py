import os
from typing import *

import torch
import torchvision
import numpy as np
from bigotis.models import TamingDecoder, Aphantasia
from loguru import logger

from server.server_config import MODEL_NAME, DEBUG, DEBUG_OUT_DIR

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

    def load_model(
        self,
        model_name: str,
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
            if self.taming_decoder is None:
                logger.info("SETTING UP TAMING...")
                self.taming_decoder = TamingDecoder()
                self.taming_decoder.eval()

            model = self.taming_decoder

        elif model_name == "aphantasia":
            if self.aphantasia is None:
                logger.info("SETTING UP APHANTASIA...")
                self.aphantasia = Aphantasia()
                self.aphantasia.eval()

            model = self.aphantasia

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
        **kwargs,
    ) -> None:
        """
        Set up optimization parameters.

        Args:
            prompt (str): prompt used to guide the optimization.
            cond_img (np.ndarray): initial image used to start the optimization.
            mask (np.ndarray): mask determining the region to optimize.
            lr (float): learning rate.
        """
        self.mask = mask.to(DEVICE)
        self.cond_img = cond_img.to(DEVICE)

        self.layer_size = mask.shape[2::]

        self.model = model_factory.load_model(MODEL_NAME)

        text_latents = self.model.get_clip_text_encodings(prompt, )
        text_latents = text_latents.detach()
        text_latents = text_latents.to(DEVICE)
        self.text_latents = text_latents

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
        num_iters: int = 32,
    ):
        for iter_idx in range(num_iters):
            gen_img = self.model.get_img_from_latents(self.gen_latents, )
            gen_img = (self.mask * gen_img) + (1 - self.mask) * self.cond_img
            
            loss = 10 * torch.nn.functional.mse_loss(gen_img, self.cond_img,)
            logger.debug(f"MSE LOSS {loss}")
        
            self.rec_optimizer.zero_grad()
            loss.backward(retain_graph=False, )
            self.rec_optimizer.step()

        return

    def optimize(self, ) -> torch.Tensor:
        """
        Perform one optimization step.

        Returns:
            torch.Tensor: updated image.
        """
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

        img_aug = self.model.augment(gen_img, )
        img_latents = self.model.get_clip_img_encodings(img_aug, ).to(DEVICE)

        loss = (self.text_latents -
                img_latents).norm(dim=-1).div(2).arcsin().pow(2).mul(2).mean()

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

        return gen_img


# NOTE: keeping this code because it includes cool stuff such as regularizers.
# class LayeredGenerator(torch.nn.Module):
#     def __init__(
#         self,
#         layer_list,
#         target_img_size=(128, 128),
#         lr: float = 0.5,
#         init_latents: torch.Tensor = None,
#         cond_img: torch.Tensor = None,
#     ):
#         super(LayeredGenerator, self).__init__()

#         self.layer_list = layer_list
#         self.target_img_size = target_img_size
#         self.lr = lr
#         self.init_latents = init_latents
#         self.cond_img = cond_img

#         self.gen_latents = None
#         self.init_img = None
#         self.reset_gen_latent()

#         self.optimizer = None
#         self.reset_optimizer()

#         self.num_iters = 0

#     def reset_gen_latent(self, ):
#         if self.gen_latents is None:
#             self.gen_latents = model.get_random_latents(
#                 target_img_height=self.target_img_size[0],
#                 target_img_width=self.target_img_size[1],
#             )
#             self.gen_latents = self.gen_latents.to(DEVICE)
#             self.gen_latents.requires_grad = True
#             self.gen_latents = torch.nn.Parameter(self.gen_latents)

#         else:
#             self.gen_latents.data = model.get_random_latents(
#                 target_img_height=self.target_img_size[0],
#                 target_img_width=self.target_img_size[1],
#             ).to(DEVICE)

#         if self.init_latents is not None:
#             self.gen_latents.data = self.init_latents

#         if self.cond_img is not None:
#             self.cond_img = self.cond_img.resize(self.target_img_size)
#             self.gen_latents.data = model.get_latents_from_img(
#                 self.cond_img, ).detach()

#         self.init_img = model.get_img_from_latents(
#             self.gen_latents).clone().detach()

#         self.init_gen_latents = self.gen_latents.clone().detach()

#     def reset_optimizer(self, ):
#         self.optimizer = torch.optim.AdamW(
#             params=[self.gen_latents],
#             lr=self.lr,
#             betas=(0.9, 0.999),
#             weight_decay=0.1,
#         )

#     def optimize(
#         self,
#         iters_per_mask: int = 1,
#     ):
#         # try:
#         loss_dict = {}
#         for layer_idx, layer in enumerate(self.layer_list):
#             logging.info(f"\nCOMPUTING LOSS OF LAYER {layer_idx}")

#             mask = layer.mask.clone()
#             mask = torch.nn.functional.interpolate(
#                 mask[None],
#                 self.target_img_size,
#                 mode="bilinear",
#             )

#             # if global_mask is None:
#             #     global_mask = mask
#             # else:
#             #     mask = mask * (1 - global_mask)
#             #     global_mask = (global_mask + mask).clip(0, 1)

#             for step in range(iters_per_mask):
#                 loss = 0

#                 img_rec = model.get_img_from_latents(self.gen_latents)

#                 # merged = img_rec * mask
#                 merged = img_rec

#                 torchvision.transforms.ToPILImage(mode='RGB')(img_rec[0]).save(
#                     f"{self.num_iters}-{layer_idx}-{step}.jpg")
#                 torchvision.transforms.ToPILImage(mode="L")(
#                     mask[0]).save("mask.jpg")
#                 torchvision.transforms.ToPILImage(mode="RGB")(
#                     merged[0]).save("masked.jpg")
#                 torchvision.transforms.ToPILImage(mode="RGB")(
#                     self.init_img[0]).save("init_img.jpg")

#                 img_rec_aug = model.augment(merged, )
#                 img_latents = model.get_clip_img_encodings(
#                     img_rec_aug, ).to(DEVICE)

#                 clip_loss = (layer.text_latents - img_latents).norm(
#                     dim=-1).div(2).arcsin().pow(2).mul(2).mean()

#                 # clip_loss = -10 * torch.cosine_similarity(
#                 #     layer.text_latents,
#                 #     img_latents,
#                 # ).mean()

#                 # rand_img_reg = 10 * torch.nn.functional.mse_loss(
#                 #     img_rec * (1 - mask),
#                 #     self.init_img * (1 - mask),
#                 # )

#                 rand_img_reg = torch.nn.functional.mse_loss(
#                     img_rec * (1 - mask),
#                     self.init_img * (1 - mask),
#                 )

#                 layer_loss = clip_loss + rand_img_reg

#                 loss += layer_loss

#                 logging.info(f"RAND REG --> {rand_img_reg}")
#                 logging.info(f"CLIP LOSS --> {clip_loss}")
#                 logging.info(f"LAYER LOSS --> {layer_loss}")
#                 logging.info(f"LOSS --> {loss} \n\n")

#                 def scale_grad(grad, ):
#                     grad_size = grad.shape[2:4]

#                     grad_mask = torch.nn.functional.interpolate(
#                         mask,
#                         grad_size,
#                         mode="bilinear",
#                     )

#                     if len(grad.shape) == 5:
#                         grad_mask = grad_mask[..., None]

#                     masked_grad = grad * grad_mask

#                     return masked_grad

#                 # hook = self.gen_latents.register_hook(scale_grad)
#                 hook_img = merged.register_hook(scale_grad)

#                 self.optimizer.zero_grad()
#                 loss.backward(retain_graph=False, )
#                 self.optimizer.step()

#                 # hook.remove()
#                 hook_img.remove()

#                 loss_dict[f"layer_{layer_idx}"] = loss

#                 with torch.no_grad():
#                     latent_mask = torch.nn.functional.interpolate(
#                         mask,
#                         self.gen_latents.shape[2:4],
#                         mode="bilinear",
#                     )

#                     if len(self.gen_latents.shape) == 5:
#                         latent_mask = latent_mask[..., None]

#                     self.gen_latents.data = self.gen_latents.data * latent_mask + self.init_gen_latents * (
#                         1 - latent_mask)

#                 self.num_iters += 1

#             self.init_img = img_rec.detach()
#             self.init_gen_latents = self.gen_latents.detach().clone()

#         # except Exception as e:
#         #     logging.info(f"XXX: ERROR IN GENERATE {e}")

#         state = None
#         return img_rec, loss_dict, state

#     def optimize_scaled(
#         self,
#         scale_factor=4,
#         num_iters=200,
#     ):
#         scaled_gen_latents = model.get_random_latents(
#             target_img_height=self.target_img_size[0] * scale_factor,
#             target_img_width=self.target_img_size[1] * scale_factor,
#         ).to(DEVICE)

#         for idx in range(num_iters):
#             x_init = random.randint(
#                 0, self.target_img_size[0] // 16 * (scale_factor - 1))
#             y_init = random.randint(
#                 0, self.target_img_size[1] // 16 * (scale_factor - 1))

#             crop_gen_latents = scaled_gen_latents[:, :, x_init:x_init +
#                                                   self.target_img_size[0] //
#                                                   16, y_init:y_init +
#                                                   self.target_img_size[1] //
#                                                   16, ]

#             self.gen_latents.data = crop_gen_latents

#             loss = 0
#             img_rec = model.get_img_from_latents(self.gen_latents)

#             torchvision.transforms.ToPILImage(mode='RGB')(
#                 img_rec[0]).save(f"{idx}-crop.jpg")
#             # torchvision.transforms.ToPILImage(mode="L")(
#             #     mask[0]).save("mask.jpg")
#             # torchvision.transforms.ToPILImage(mode="RGB")(
#             #     merged[0]).save("masked.jpg")
#             # torchvision.transforms.ToPILImage(mode="RGB")(
#             #     self.init_img[0]).save("init_img.jpg")

#             img_rec_aug = model.augment(img_rec, )
#             img_latents = model.get_clip_img_encodings(
#                 img_rec_aug, ).to(DEVICE)

#             # clip_loss = (self.layer_list[0].text_latents - img_latents).norm(
#             #     dim=-1).div(2).arcsin().pow(2).mul(2).mean()

#             clip_loss = -10 * torch.cosine_similarity(
#                 self.layer_list[0].text_latents,
#                 img_latents,
#             ).mean()

#             # rand_img_reg = 10 * torch.nn.functional.mse_loss(
#             #     img_rec * (1 - mask),
#             #     self.init_img * (1 - mask),
#             # )

#             # rand_img_reg = torch.nn.functional.mse_loss(
#             #     img_rec * (1 - mask),
#             #     self.init_img * (1 - mask),
#             # )

#             # layer_loss = clip_loss + rand_img_reg

#             loss += clip_loss

#             # logging.info(f"RAND REG --> {rand_img_reg}")
#             # logging.info(f"CLIP LOSS --> {clip_loss}")
#             # logging.info(f"LAYER LOSS --> {layer_loss}")
#             logging.info(f"LOSS --> {loss} \n\n")

#             self.optimizer.zero_grad()
#             loss.backward(retain_graph=False, )
#             self.optimizer.step()

#             scaled_gen_latents[:, :,
#                                x_init:x_init + self.target_img_size[0] // 16,
#                                y_init:y_init + self.target_img_size[1] //
#                                16, ] = self.gen_latents.data.clone().detach()

#             final_img = torch.zeros(
#                 (1, 3, self.target_img_size[0] * scale_factor,
#                  self.target_img_size[1] * scale_factor))

#             for y_scale in range(scale_factor):
#                 for x_scale in range(scale_factor):
#                     img_y = self.target_img_size[0]
#                     img_x = self.target_img_size[1]
#                     embed_y = img_y // 16
#                     embed_x = img_x // 16

#                     with torch.no_grad():
#                         final_img[:, :, img_y * y_scale:img_y * (y_scale + 1),
#                                   img_x * x_scale:img_x *
#                                   (x_scale + 1)] = model.get_img_from_latents(
#                                       scaled_gen_latents[:, :, embed_y *
#                                                          y_scale:embed_y *
#                                                          (y_scale + 1),
#                                                          embed_x *
#                                                          x_scale:embed_x *
#                                                          (x_scale + 1)])
#             final_img_pil = torchvision.transforms.ToPILImage()(final_img[0])
#             final_img_pil.save(f"{idx}.png")

#         return img_rec, None, None