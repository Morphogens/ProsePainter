import time

import torch
import numpy as np
from PIL import Image
from bigotis.models import TamingDecoder
from PIL import Image

import webserver

target_img_size = 256

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

taming_decoder = TamingDecoder()
model = taming_decoder.to(device)


class LayerLoss:
    def __init__(
        self,
        layer: webserver.Layer,
    ):
        self.text_emb = taming_decoder.get_clip_text_encodings(layer.prompt, )

        # get alpha mask
        mask = torch.from_numpy(layer.img[:, :, -1]).to(device)
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
        merged = image * mask  #+ image.detach() * (1-mask)
        cutouts = taming_decoder.augment(merged, )
        image_emb = taming_decoder.get_clip_img_encodings(cutouts, )

        dists = image_emb.sub(
            self.text_emb, ).norm(dim=2).div(2).arcsin().pow(2).mul(2)

        return dists.mean()


class UserGuide:
    user_session: webserver.UserSession

    def __init__(self, ):
        self.last_prompts = []
        self.pMs = []
        self.user_session = None
        self.z = None
        self.layers_loss = None
        self.user_layers = None

    def init(self, z):
        self.z = z

    def update_state(
        self,
        user_session: webserver.UserSession,
    ):
        self.user_session = user_session
        if self.user_layers is not user_session.layers:
            self.user_layers = user_session.layers
            self.layers_loss = [
                LayerLoss(layer) for layer in user_session.layers
            ]

    def apply_losses(
        self,
        z,
        rec_image,
    ):
        if not self.user_session:
            return 0

        loss = 0
        for layer_idx, layer_loss in enumerate(self.layers_loss):

            def scale_grad(grad):
                N, C, H, W = grad.shape
                m = layer_loss.mask.clone()
                for covering in self.layers_loss[layer_idx + 1:]:
                    m -= covering.mask
                    m.clamp_(0, 1)

                return grad * torch.nn.functional.interpolate(
                    m[None, None], (H, W))

            hook = z.register_hook(scale_grad)
            loss = layer_loss(rec_image, )
            loss.backward(retain_graph=True, )
            hook.remove()


#         for pm, layer in zip(pms, layers):
#             loss = pm(iii)
#             total += loss

#             def scale_grad(grad):
#                 global MM
#                 N, C, H, W = grad.shape
#                 radius = 0.25
#                 m = create_circular_mask(H, W, [x * W, y * H], radius=radius*W)
#                 MM = m
#                 m = torch.from_numpy(m).to(device)

#                 return grad * m
#             # hook = self.z.register_hook(scale_grad)

#             # SLOW! requires a full backwards pass on CLIP image + text nets
#             # if we can collapse it to one backwards (mask?) it will speed things up dramatically
#             # We do this so we can scale the gradients WRT per-anchor location
#             loss.backward(retain_graph=True)
#             # hook.remove()

# #         total.backward(retain_graph=True)

        return

    # @torch.no_grad()
    # def _get_prompts(
    #     self,
    #     prompts,
    # ):
    #     if prompts == self.last_prompts:
    #         return self.pMs

    #     self.pMs = []

    #     self.last_prompts = prompts
    #     for prompt in prompts:
    #         print("Encoding", prompt)
    #         embed = taming_decoder.get_clip_text_encodings(prompt)
    #         self.pMs.append(Prompt(embed).to(device))

    #     return self.pMs


def create_circular_mask(
    h,
    w,
    center=None,
    radius=None,
):
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))

    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)

    mask = dist_from_center <= radius

    return mask


# import matplotlib.pyplot as plt

# H = 400
# W = 800

# m = create_circular_mask(H, W, [0.1 * W, 0.1 * H], radius=0.1 * W)
# plt.imshow(m)

NUM_STEPS = 100
LR = 0.5

z_logits = taming_decoder.get_random_z(
    target_img_height=target_img_size,
    target_img_width=target_img_size,
)

user_guided = UserGuide()
user_guided.init(z_logits)

z_logits = torch.nn.Parameter(z_logits)

optimizer = torch.optim.AdamW(
    params=[z_logits],
    lr=LR,
    betas=(0.9, 0.999),
    weight_decay=0.1,
)
gen_img_list = []
z_logits_list = []

try:
    while not webserver.us or not webserver.us.state.get("run"):
        time.sleep(0.1)

        if webserver.us:
            user_guided.update_state(webserver.us)

            x_rec = taming_decoder.get_img_rec_from_z(z_logits)
            user_guided.apply_losses(
                z=z_logits,
                rec_image=x_rec,
            )

            webserver.process_step(x_rec.cpu(), )

        else:
            continue
            # user_guided.update_state({})

except KeyboardInterrupt:
    pass
