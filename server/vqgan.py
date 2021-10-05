import time
import logging

import torch
import numpy as np
from PIL import Image
from bigotis.models import TamingDecoder
from PIL import Image

# import webserver

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("DEVICE: ", device)

taming_decoder = TamingDecoder()
model = taming_decoder.to(device)


class LayerLoss:
    def __init__(
        self,
        layer,
    ):
        self.text_emb = taming_decoder.get_clip_text_encodings(layer.prompt, )
        self.text_emb = self.text_emb.detach()
        self.text_emb = self.text_emb.to(device)

        # get alpha mask
        # XXX: getting the first channel to create the mask, not the last one
        mask = torch.from_numpy(layer.img[:, :, 0]).to(device)
        mask[mask > 0] = 1
        mask = mask.float()
        self.mask = mask

    def __call__(
        self,
        image,
    ):
        N, C, H, W = image.shape
        mask = torch.nn.functional.interpolate(
            self.mask[None, None],
            (H, W),
            mode="bilinear",
        )
        # merged = image * mask  #+ image.detach() * (1-mask)
        merged = image
        self.merged = merged
        import torchvision
        torchvision.transforms.ToPILImage(mode="L")(mask[0]).save("mask.jpg")
        torchvision.transforms.ToPILImage(mode="RGB")(
            merged[0]).save("masked.jpg")
        cutouts = taming_decoder.augment(merged, )
        image_emb = taming_decoder.get_clip_img_encodings(cutouts, )
        image_emb = image_emb.to(device)

        # dist = image_emb.sub(
        #     self.text_emb, ).norm(dim=2).div(2).arcsin().pow(2).mul(2)

        dist = -10 * torch.cosine_similarity(self.text_emb, image_emb)

        return dist.mean()


class LayeredGenerator(torch.nn.Module):
    def __init__(
        self,
        layer_list,
        target_img_size=128,
        lr: float = 0.5,
    ):
        super(LayeredGenerator, self).__init__()

        self.lr = lr
        self.target_img_size = target_img_size

        self.layer_loss_list = None
        self.reset_layers(layer_list, )

        self.gen_latents = None
        self.reset_gen_latent()

        self.optimizer = None
        self.reset_optimizer()

    def reset_layers(
        self,
        layer_list,
    ):
        self.layer_loss_list = [LayerLoss(layer) for layer in layer_list]

    def reset_gen_latent(self, ):
        self.gen_latents = taming_decoder.get_random_latents(
            target_img_height=self.target_img_size,
            target_img_width=self.target_img_size,
        )
        self.gen_latents = self.gen_latents.to(device)
        self.gen_latents.requires_grad = True
        self.gen_latents = torch.nn.Parameter(self.gen_latents)

    def reset_optimizer(self, ):
        self.optimizer = torch.optim.AdamW(
            params=[self.gen_latents],
            lr=self.lr,
            betas=(0.9, 0.999),
            weight_decay=0.1,
        )

    def optimize(self, ):
        try:
            x_rec = taming_decoder.get_img_from_latents(self.gen_latents)

            loss = 0
            loss_dict = {}
            for layer_idx, layer_loss in enumerate(self.layer_loss_list):
                logging.info(f"COMPUTING LOSS OF LAYER {layer_idx}")

                def scale_grad(grad):
                    print("GRAD SHAPE", grad.shape)
                    N, C, H, W = grad.shape
                    mask = layer_loss.mask.clone()
                    # for covering in self.layer_loss_list[layer_idx]:
                    #     mask -= covering.mask
                    #     mask.clamp_(0, 1)

                    mask = torch.nn.functional.interpolate(
                        mask[None, None],
                        (H, W),
                    )

                    masked_grad = grad * mask

                    return masked_grad

                loss = layer_loss(x_rec, )

                hook = self.gen_latents.register_hook(scale_grad)
                hook_img = layer_loss.merged.register_hook(scale_grad)

                self.optimizer.zero_grad()
                loss.backward(retain_graph=False, )
                self.optimizer.step()

                hook.remove()
                hook_img.remove()

                logging.info(f"LOSS {loss}")
                loss_dict[f"layer_{layer_idx}"] = loss

        except Exception as e:
            logging.info(f"XXX: ERROR IN GENERATE {e}")

        state = None
        return x_rec, loss_dict, state


if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class Layer:
        color: str
        strength: int
        prompt: str
        img: np.ndarray

    layer = Layer(
        color=000,
        strength=1,
        prompt="a pink dog",
        img=np.zeros((100, 100, 3)),
    )
    layer_list = [layer]

    layered_generator = LayeredGenerator(layer_list, )
    out = layered_generator()
