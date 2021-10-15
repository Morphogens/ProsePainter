import random
import time
import logging

import torch
import torchvision
import numpy as np
from PIL import Image
from bigotis.models import TamingDecoder, Aphantasia
from torchvision.transforms.functional import scale

torch.manual_seed(123)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("DEVICE: ", DEVICE)

taming_decoder = TamingDecoder()
taming_decoder.eval()
model = taming_decoder.to(DEVICE)

# aphantasia = Aphantasia()
# aphantasia.eval()
# model = aphantasia


class Layer:
    def __init__(
        self,
        prompt: str,
        mask: np.ndarray,
        cond_img: np.ndarray,
        lr: float,
        **kwargs,
    ):
        self.mask = mask.to(DEVICE)
        self.cond_img = cond_img.to(DEVICE)

        text_latents = model.get_clip_text_encodings(prompt, )
        text_latents = text_latents.detach()
        text_latents = text_latents.to(DEVICE)
        self.text_latents = text_latents

        self.gen_latents = model.get_latents_from_img(cond_img, )
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

        return

    def optimize(self, ):
        gen_img = model.get_img_from_latents(self.gen_latents, )

        torchvision.transforms.ToPILImage(mode="L")(
            self.mask[0]).save("mask.jpg")
        torchvision.transforms.ToPILImage(mode="RGB")(
            gen_img[0], ).save("gen_img.jpg")
        torchvision.transforms.ToPILImage(mode="RGB")(
            self.cond_img[0]).save("init_img.jpg")

        img_aug = model.augment(gen_img, )
        img_latents = model.get_clip_img_encodings(img_aug, ).to(DEVICE)

        loss = (self.text_latents -
                img_latents).norm(dim=-1).div(2).arcsin().pow(2).mul(2).mean()

        # loss = -10 * torch.cosine_similarity(
        #     self.text_latents,
        #     img_latents,
        # ).mean()

        logging.info(f"LOSS --> {loss} \n\n")

        def scale_grad(grad, ):
            grad_size = grad.shape[2:4]

            grad_mask = torch.nn.functional.interpolate(
                self.mask,
                grad_size,
                mode="bilinear",
            )

            if len(grad.shape) == 5:
                grad_mask = grad_mask[..., None]

            masked_grad = grad * grad_mask

            return masked_grad

        # hook = self.gen_latents.register_hook(scale_grad)
        hook_img = gen_img.register_hook(scale_grad)

        self.optimizer.zero_grad()
        loss.backward(retain_graph=False, )
        self.optimizer.step()

        # hook.remove()
        hook_img.remove()

        return gen_img


class LayeredGenerator(torch.nn.Module):
    def __init__(
        self,
        layer_list,
        target_img_size=(128, 128),
        lr: float = 0.5,
        init_latents: torch.Tensor = None,
        cond_img: torch.Tensor = None,
    ):
        super(LayeredGenerator, self).__init__()

        self.layer_list = layer_list
        self.target_img_size = target_img_size
        self.lr = lr
        self.init_latents = init_latents
        self.cond_img = cond_img

        self.gen_latents = None
        self.init_img = None
        self.reset_gen_latent()

        self.optimizer = None
        self.reset_optimizer()

        self.num_iters = 0

    def reset_gen_latent(self, ):
        if self.gen_latents is None:
            self.gen_latents = model.get_random_latents(
                target_img_height=self.target_img_size[0],
                target_img_width=self.target_img_size[1],
            )
            self.gen_latents = self.gen_latents.to(DEVICE)
            self.gen_latents.requires_grad = True
            self.gen_latents = torch.nn.Parameter(self.gen_latents)

        else:
            self.gen_latents.data = model.get_random_latents(
                target_img_height=self.target_img_size[0],
                target_img_width=self.target_img_size[1],
            ).to(DEVICE)

        if self.init_latents is not None:
            self.gen_latents.data = self.init_latents

        if self.cond_img is not None:
            self.cond_img = self.cond_img.resize(self.target_img_size)
            self.gen_latents.data = model.get_latents_from_img(
                self.cond_img, ).detach()

        self.init_img = model.get_img_from_latents(
            self.gen_latents).clone().detach()

        self.init_gen_latents = self.gen_latents.clone().detach()

    def reset_optimizer(self, ):
        self.optimizer = torch.optim.AdamW(
            params=[self.gen_latents],
            lr=self.lr,
            betas=(0.9, 0.999),
            weight_decay=0.1,
        )

    def optimize(
        self,
        iters_per_mask: int = 1,
    ):
        # try:
        loss_dict = {}
        for layer_idx, layer in enumerate(self.layer_list):
            logging.info(f"\nCOMPUTING LOSS OF LAYER {layer_idx}")

            mask = layer.mask.clone()
            mask = torch.nn.functional.interpolate(
                mask[None],
                self.target_img_size,
                mode="bilinear",
            )

            # if global_mask is None:
            #     global_mask = mask
            # else:
            #     mask = mask * (1 - global_mask)
            #     global_mask = (global_mask + mask).clip(0, 1)

            for step in range(iters_per_mask):
                loss = 0

                img_rec = model.get_img_from_latents(self.gen_latents)

                # merged = img_rec * mask
                merged = img_rec

                torchvision.transforms.ToPILImage(mode='RGB')(img_rec[0]).save(
                    f"{self.num_iters}-{layer_idx}-{step}.jpg")
                torchvision.transforms.ToPILImage(mode="L")(
                    mask[0]).save("mask.jpg")
                torchvision.transforms.ToPILImage(mode="RGB")(
                    merged[0]).save("masked.jpg")
                torchvision.transforms.ToPILImage(mode="RGB")(
                    self.init_img[0]).save("init_img.jpg")

                img_rec_aug = model.augment(merged, )
                img_latents = model.get_clip_img_encodings(
                    img_rec_aug, ).to(DEVICE)

                clip_loss = (layer.text_latents - img_latents).norm(
                    dim=-1).div(2).arcsin().pow(2).mul(2).mean()

                # clip_loss = -10 * torch.cosine_similarity(
                #     layer.text_latents,
                #     img_latents,
                # ).mean()

                # rand_img_reg = 10 * torch.nn.functional.mse_loss(
                #     img_rec * (1 - mask),
                #     self.init_img * (1 - mask),
                # )

                rand_img_reg = torch.nn.functional.mse_loss(
                    img_rec * (1 - mask),
                    self.init_img * (1 - mask),
                )

                layer_loss = clip_loss + rand_img_reg

                loss += layer_loss

                logging.info(f"RAND REG --> {rand_img_reg}")
                logging.info(f"CLIP LOSS --> {clip_loss}")
                logging.info(f"LAYER LOSS --> {layer_loss}")
                logging.info(f"LOSS --> {loss} \n\n")

                def scale_grad(grad, ):
                    grad_size = grad.shape[2:4]

                    grad_mask = torch.nn.functional.interpolate(
                        mask,
                        grad_size,
                        mode="bilinear",
                    )

                    if len(grad.shape) == 5:
                        grad_mask = grad_mask[..., None]

                    masked_grad = grad * grad_mask

                    return masked_grad

                # hook = self.gen_latents.register_hook(scale_grad)
                hook_img = merged.register_hook(scale_grad)

                self.optimizer.zero_grad()
                loss.backward(retain_graph=False, )
                self.optimizer.step()

                # hook.remove()
                hook_img.remove()

                loss_dict[f"layer_{layer_idx}"] = loss

                with torch.no_grad():
                    latent_mask = torch.nn.functional.interpolate(
                        mask,
                        self.gen_latents.shape[2:4],
                        mode="bilinear",
                    )

                    if len(self.gen_latents.shape) == 5:
                        latent_mask = latent_mask[..., None]

                    self.gen_latents.data = self.gen_latents.data * latent_mask + self.init_gen_latents * (
                        1 - latent_mask)

                self.num_iters += 1

            self.init_img = img_rec.detach()
            self.init_gen_latents = self.gen_latents.detach().clone()

        # except Exception as e:
        #     logging.info(f"XXX: ERROR IN GENERATE {e}")

        state = None
        return img_rec, loss_dict, state

    def optimize_scaled(
        self,
        scale_factor=4,
        num_iters=200,
    ):
        scaled_gen_latents = model.get_random_latents(
            target_img_height=self.target_img_size[0] * scale_factor,
            target_img_width=self.target_img_size[1] * scale_factor,
        ).to(DEVICE)

        for idx in range(num_iters):
            x_init = random.randint(
                0, self.target_img_size[0] // 16 * (scale_factor - 1))
            y_init = random.randint(
                0, self.target_img_size[1] // 16 * (scale_factor - 1))

            crop_gen_latents = scaled_gen_latents[:, :, x_init:x_init +
                                                  self.target_img_size[0] //
                                                  16, y_init:y_init +
                                                  self.target_img_size[1] //
                                                  16, ]

            self.gen_latents.data = crop_gen_latents

            loss = 0
            img_rec = model.get_img_from_latents(self.gen_latents)

            torchvision.transforms.ToPILImage(mode='RGB')(
                img_rec[0]).save(f"{idx}-crop.jpg")
            # torchvision.transforms.ToPILImage(mode="L")(
            #     mask[0]).save("mask.jpg")
            # torchvision.transforms.ToPILImage(mode="RGB")(
            #     merged[0]).save("masked.jpg")
            # torchvision.transforms.ToPILImage(mode="RGB")(
            #     self.init_img[0]).save("init_img.jpg")

            img_rec_aug = model.augment(img_rec, )
            img_latents = model.get_clip_img_encodings(
                img_rec_aug, ).to(DEVICE)

            # clip_loss = (self.layer_list[0].text_latents - img_latents).norm(
            #     dim=-1).div(2).arcsin().pow(2).mul(2).mean()

            clip_loss = -10 * torch.cosine_similarity(
                self.layer_list[0].text_latents,
                img_latents,
            ).mean()

            # rand_img_reg = 10 * torch.nn.functional.mse_loss(
            #     img_rec * (1 - mask),
            #     self.init_img * (1 - mask),
            # )

            # rand_img_reg = torch.nn.functional.mse_loss(
            #     img_rec * (1 - mask),
            #     self.init_img * (1 - mask),
            # )

            # layer_loss = clip_loss + rand_img_reg

            loss += clip_loss

            # logging.info(f"RAND REG --> {rand_img_reg}")
            # logging.info(f"CLIP LOSS --> {clip_loss}")
            # logging.info(f"LAYER LOSS --> {layer_loss}")
            logging.info(f"LOSS --> {loss} \n\n")

            self.optimizer.zero_grad()
            loss.backward(retain_graph=False, )
            self.optimizer.step()

            scaled_gen_latents[:, :,
                               x_init:x_init + self.target_img_size[0] // 16,
                               y_init:y_init + self.target_img_size[1] //
                               16, ] = self.gen_latents.data.clone().detach()

            final_img = torch.zeros(
                (1, 3, self.target_img_size[0] * scale_factor,
                 self.target_img_size[1] * scale_factor))

            for y_scale in range(scale_factor):
                for x_scale in range(scale_factor):
                    img_y = self.target_img_size[0]
                    img_x = self.target_img_size[1]
                    embed_y = img_y // 16
                    embed_x = img_x // 16

                    with torch.no_grad():
                        final_img[:, :, img_y * y_scale:img_y * (y_scale + 1),
                                  img_x * x_scale:img_x *
                                  (x_scale + 1)] = model.get_img_from_latents(
                                      scaled_gen_latents[:, :, embed_y *
                                                         y_scale:embed_y *
                                                         (y_scale + 1),
                                                         embed_x *
                                                         x_scale:embed_x *
                                                         (x_scale + 1)])
            final_img_pil = torchvision.transforms.ToPILImage()(final_img[0])
            final_img_pil.save(f"{idx}.png")

        return img_rec, None, None