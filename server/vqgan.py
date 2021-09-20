#!/usr/bin/env python
# coding: utf-8

# # Generates images from text prompts with VQGAN and CLIP (z+quantize method).
# 
# By jbustter https://twitter.com/jbusted1 .
# Based on a notebook by Katherine Crowson (https://github.com/crowsonkb, https://twitter.com/RiversHaveWings)
# 


import webserver
import asyncio
from threading import Thread
loop = asyncio.get_event_loop()

def mm_runner():
#     asyncio.set_event_loop(loop)
    webserver.main()
    loop.run_forever()
    
mm_thread = Thread(target=mm_runner, daemon=True)
mm_thread.start()


# In[3]:


import torch
from PIL import Image
x = torch.randn(128, 128, 3)
webserver.process_step(x)


# In[4]:


# !git clone https://github.com/openai/CLIP
# !git clone https://github.com/CompVis/taming-transformers
# !pip install ftfy regex tqdm omegaconf pytorch-lightning
# # !pip install kornia
# !pip install einops


# In[5]:



# !curl -L 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1' > vqgan_imagenet_f16_16384.yaml
# !curl -L 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1' > vqgan_imagenet_f16_16384.ckpt


# In[6]:


import argparse
import math
from pathlib import Path
import sys

sys.path.append('./taming-transformers')

from IPython import display
from omegaconf import OmegaConf
from PIL import Image
from taming.models import cond_transformer, vqgan
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.notebook import tqdm
import numpy as np

from CLIP import clip

import kornia.augmentation as K


# In[ ]:





# In[ ]:





# In[7]:


def noise_gen(shape):
    n, c, h, w = shape
    noise = torch.zeros([n, c, 1, 1])
    for i in reversed(range(5)):
        h_cur, w_cur = h // 2**i, w // 2**i
        noise = F.interpolate(noise, (h_cur, w_cur), mode='bicubic', align_corners=False)
        noise += torch.randn([n, c, h_cur, w_cur]) / 5
    return noise


def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x/a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]


def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size

    input = input.view([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.view([n, c, h, w])
    return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)
    

# def replace_grad(fake, real):
#     return fake.detach() - real.detach() + real


class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)


class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None

replace_grad = ReplaceGrad.apply

clamp_with_grad = ClampWithGrad.apply
# clamp_with_grad = torch.clamp

def vector_quantize(x, codebook):
    d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return replace_grad(x_q, x)


class Prompt(nn.Module):
    def __init__(self, embed, weight=1., stop=float('-inf')):
        super().__init__()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))

    def forward(self, input):
        
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize((self.embed).unsqueeze(0), dim=2)

        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        return self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop)).mean()


def parse_prompt(prompt):
    vals = prompt.rsplit(':', 2)
    vals = vals + ['', '1', '-inf'][len(vals):]
    return vals[0], float(vals[1]), float(vals[2])

def one_sided_clip_loss(input, target, labels=None, logit_scale=100):
    input_normed = F.normalize(input, dim=-1)
    target_normed = F.normalize(target, dim=-1)
    logits = input_normed @ target_normed.T * logit_scale
    if labels is None:
        labels = torch.arange(len(input), device=logits.device)
    return F.cross_entropy(logits, labels)

class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

        self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))

    def set_cut_pow(self, cut_pow):
      self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        cutouts_full = []
        
        min_size_width = min(sideX, sideY)
        lower_bound = float(self.cut_size/min_size_width)
        
        for ii in range(self.cutn):
            
            
          size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)

          offsetx = torch.randint(0, sideX - size + 1, ())
          offsety = torch.randint(0, sideY - size + 1, ())
          cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
          cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))

        
        cutouts = torch.cat(cutouts, dim=0)

        if args.use_augs:
          cutouts = augs(cutouts)

        if args.noise_fac:
          facs = cutouts.new_empty([cutouts.shape[0], 1, 1, 1]).uniform_(0, args.noise_fac)
          cutouts = cutouts + facs * torch.randn_like(cutouts)
        

        return clamp_with_grad(cutouts, 0, 1)


def load_vqgan_model(config_path, checkpoint_path):
    config = OmegaConf.load(config_path)
    if config.model.target == 'taming.models.vqgan.VQModel':
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    elif config.model.target == 'taming.models.vqgan.GumbelVQ':
        model = vqgan.GumbelVQ(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    else:
        raise ValueError(f'unknown model type: {config.model.target}')
    del model.loss
    return model

def resize_image(image, out_size):
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio)**0.5), round((area / ratio)**0.5)
    return image.resize(size, Image.LANCZOS)

class TVLoss(nn.Module):
    def forward(self, input):
        input = F.pad(input, (0, 1, 0, 1), 'replicate')
        x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
        y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
        diff = x_diff**2 + y_diff**2 + 1e-8
        return diff.mean(dim=1).sqrt().mean()

class GaussianBlur2d(nn.Module):
    def __init__(self, sigma, window=0, mode='reflect', value=0):
        super().__init__()
        self.mode = mode
        self.value = value
        if not window:
            window = max(math.ceil((sigma * 6 + 1) / 2) * 2 - 1, 3)
        if sigma:
            kernel = torch.exp(-(torch.arange(window) - window // 2)**2 / 2 / sigma**2)
            kernel /= kernel.sum()
        else:
            kernel = torch.ones([1])
        self.register_buffer('kernel', kernel)

    def forward(self, input):
        n, c, h, w = input.shape
        input = input.view([n * c, 1, h, w])
        start_pad = (self.kernel.shape[0] - 1) // 2
        end_pad = self.kernel.shape[0] // 2
        input = F.pad(input, (start_pad, end_pad, start_pad, end_pad), self.mode, self.value)
        input = F.conv2d(input, self.kernel[None, None, None, :])
        input = F.conv2d(input, self.kernel[None, None, :, None])
        return input.view([n, c, h, w])

class EMATensor(nn.Module):
    """implmeneted by Katherine Crowson"""
    def __init__(self, tensor, decay):
        super().__init__()
        self.tensor = nn.Parameter(tensor)
        self.register_buffer('biased', torch.zeros_like(tensor))
        self.register_buffer('average', torch.zeros_like(tensor))
        self.decay = decay
        self.register_buffer('accum', torch.tensor(1.))
        self.update()
    
    @torch.no_grad()
    def update(self):
        if not self.training:
            raise RuntimeError('update() should only be called during training')

        self.accum *= self.decay
        self.biased.mul_(self.decay)
        self.biased.add_((1 - self.decay) * self.tensor)
        self.average.copy_(self.biased)
        self.average.div_(1 - self.accum)

    def forward(self):
        if self.training:
            return self.tensor
        return self.average

get_ipython().run_line_magic('mkdir', '-p ./content/vids')


# # ARGS

# In[8]:


args = argparse.Namespace(
    
    prompts=["a photo-realistic and beautiful painting of an old man sitting on a chair next to a giant iceberg and looking at a green lush landscape."],
    size=[640//2, 512//2], 
    init_image= None,
    init_weight= 0.5,

    # clip model settings
    clip_model='ViT-B/32',
    vqgan_config='vqgan_imagenet_f16_16384.yaml',         
    vqgan_checkpoint='vqgan_imagenet_f16_16384.ckpt',
    step_size=0.1,
    
    # cutouts / crops
    cutn=64,
    cut_pow=1,

    # display
    display_freq=25,
    seed=158758,
    use_augs = True,
    noise_fac= 0.1,

    record_generation=True,

    # noise and other constraints
    use_noise = None,
    constraint_regions = False,#
    
    
    # add noise to embedding
    noise_prompt_weights = None,
    noise_prompt_seeds = [14575],#

    # mse settings
    mse_withzeros = True,
    mse_decay_rate = 50,
    mse_epoches = 5,

    # end itteration
    max_itter = -1,
)

mse_decay = 0
if args.init_weight:
  mse_decay = args.init_weight / args.mse_epoches

# <AUGMENTATIONS>
augs = nn.Sequential(
    
    K.RandomHorizontalFlip(p=0.5),
    K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode='border'), # padding_mode=2
    K.RandomPerspective(0.2,p=0.4, ),
    K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),

    )

noise = noise_gen([1, 3, args.size[0], args.size[1]])
image = TF.to_pil_image(noise.div(5).add(0.5).clamp(0, 1)[0])
image.save('init3.png')


# ### Actually do the run...

# In[9]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print('Using device:', device)
print('using prompts: ', args.prompts)

tv_loss = TVLoss() 

model = load_vqgan_model(args.vqgan_config, args.vqgan_checkpoint).to(device)
perceptor = clip.load(args.clip_model, jit=False)[0].eval().requires_grad_(False).to(device)


# In[10]:


MM = None
class UserGuide:
    def __init__(self):
        self.last_prompts = []
        self.pMs = []
        self.state = {}
        self.optimizers = {}
        self.z = None
        
    def update_state(self, state):
        self.state = state
        
    def apply_losses(self, iii):
        anchors = self.state.get("anchors", [])
        pms = self._get_prompts([x["text"] for x in anchors])
        total = 0
        for pm, anc in zip(pms, anchors):
            loss = pm(iii)
            total += loss
            x, y = anc["pos"]
            
            def scale_grad(grad):
                global MM
                N, C, H, W = grad.shape
                radius = 0.25
                m = create_circular_mask(H, W, [x * W, y * H], radius=radius*W)
                MM = m
                m = torch.from_numpy(m).to(device)
                
                
                return grad * m
            hook = self.z.register_hook(scale_grad)
            
            # SLOW! requires a full backwards pass on CLIP image + text nets
            # if we can collapse it to one backwards (mask?) it will speed things up dramatically
            # We do this so we can scale the gradients WRT per-anchor location
            loss.backward(retain_graph=True)
            hook.remove()
            
#         total.backward(retain_graph=True)
        
    def init(self, z):
        self.z = z
        

    @torch.no_grad()
    def _get_prompts(self, prompts):
        if prompts == self.last_prompts:
            return self.pMs
        self.pMs = []
        
        self.last_prompts = prompts
#         import ipdb; ipdb.set_trace()
        for prompt in prompts:
            print("Encoding", prompt)
            txt, weight, stop = parse_prompt(prompt)
            embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
            self.pMs.append(Prompt(embed, weight, stop).to(device))
        
        return self.pMs
user_guided = UserGuide()


# In[11]:


def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


# In[12]:


import matplotlib.pyplot as plt
H = 400
W = 800

m = create_circular_mask(H, W, [0.1 * W, 0.1 * H], radius=0.1*W)
plt.imshow(m)


# In[ ]:





# In[13]:


get_ipython().run_line_magic('timeit', 'm = create_circular_mask(H, W, [0.1 * W, 0.1 * H], radius=0.1*W)')


# In[14]:


mse_weight = args.init_weight

cut_size = perceptor.visual.input_resolution
# e_dim = model.quantize.e_dim

if args.vqgan_checkpoint == 'vqgan_openimages_f16_8192.ckpt':
    e_dim = 256
    n_toks = model.quantize.n_embed
    z_min = model.quantize.embed.weight.min(dim=0).values[None, :, None, None]
    z_max = model.quantize.embed.weight.max(dim=0).values[None, :, None, None]
else:
    e_dim = model.quantize.e_dim
    n_toks = model.quantize.n_e
    z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
    z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]


make_cutouts = MakeCutouts(cut_size, args.cutn, cut_pow=args.cut_pow)

f = 2**(model.decoder.num_resolutions - 1)
toksX, toksY = args.size[0] // f, args.size[1] // f

if args.seed is not None:
    torch.manual_seed(args.seed)

if args.init_image:
    pil_image = Image.open(args.init_image).convert('RGB')
    pil_image = pil_image.resize((toksX * 16, toksY * 16), Image.LANCZOS)
    pil_image = TF.to_tensor(pil_image)
    if args.use_noise:
      pil_image = pil_image + args.use_noise * torch.randn_like(pil_image) 
    z, *_ = model.encode(pil_image.to(device).unsqueeze(0) * 2 - 1)

else:
    
    one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=device), n_toks).float()

    if args.vqgan_checkpoint == 'vqgan_openimages_f16_8192.ckpt':
        z = one_hot @ model.quantize.embed.weight
    else:
        z = one_hot @ model.quantize.embedding.weight
    z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)

if args.mse_withzeros and not args.init_image:
  z_orig = torch.zeros_like(z)
else:
  z_orig = z.clone()

z.requires_grad = True

opt = optim.Adam([z], lr=args.step_size, weight_decay=0.00000000)

normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])

pMs = []

if args.noise_prompt_weights and args.noise_prompt_seeds:
  for seed, weight in zip(args.noise_prompt_seeds, args.noise_prompt_weights):
    gen = torch.Generator().manual_seed(seed)
    embed = torch.empty([1, perceptor.visual.output_dim]).normal_(generator=gen)
    pMs.append(Prompt(embed, weight).to(device))

# for prompt in args.prompts:
#     txt, weight, stop = parse_prompt(prompt)
#     embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
#     pMs.append(Prompt(embed, weight, stop).to(device))


def synth(z, quantize=True):
    if args.constraint_regions:
      z = replace_grad(z, z * z_mask)

    if quantize:
      if args.vqgan_checkpoint == 'vqgan_openimages_f16_8192.ckpt':
        z_q = vector_quantize(z.movedim(1, 3), model.quantize.embed.weight).movedim(3, 1)
      else:
        z_q = vector_quantize(z.movedim(1, 3), model.quantize.embedding.weight).movedim(3, 1)

    else:
      z_q = z.model

    return clamp_with_grad(model.decode(z_q).add(1).div(2), 0, 1)

@torch.no_grad()
def checkin(i, losses):
    losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
    tqdm.write(f'i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}')
#     out = synth(z, True)

#     TF.to_pil_image(out[0].cpu()).save('progress.png')   
#     display.display(display.Image('progress.png')) 

def ascend_txt():
    global mse_weight

    opt.zero_grad()
    out = synth(z)
    
#     with torch.no_grad():
#         out_a = synth(z, quantize=True)
#         webserver.process_step(out_a.cpu())

#     if args.record_generation:
#       with torch.no_grad():
#         global vid_index
#         out_a = synth(z, quantize=True)
#         TF.to_pil_image(out_a[0].cpu()).save(f'./content/vids/{vid_index}.png')
#         vid_index += 1
    webserver.process_step(out.cpu())
    cutouts = make_cutouts(out)

    iii = perceptor.encode_image(normalize(cutouts)).float()

    losses = []

    if args.init_weight:
        
        global z_orig
        
        losses.append(F.mse_loss(z, z_orig) * mse_weight / 2)
        # losses.append(F.mse_loss(z, z_orig) * ((1/torch.tensor((i)*2 + 1))*mse_weight) / 2)

        with torch.no_grad():
          if i > 0 and i%args.mse_decay_rate==0 and i <= args.mse_decay_rate*args.mse_epoches:

            if mse_weight - mse_decay > 0 and mse_weight - mse_decay >= mse_decay:
              mse_weight = mse_weight - mse_decay
              print(f"updated mse weight: {mse_weight}")
            else:
              mse_weight = 0
              print(f"updated mse weight: {mse_weight}")
    
    
    for prompt in pMs:
        losses.append(prompt(iii))
    
    if webserver.us:
        user_guided.update_state(webserver.us.state)
        user_guided.apply_losses(iii)
    else:
        user_guided.update({})
        
        
    loss = sum(losses)
    loss.backward()
    opt.step()
    
    if i % args.display_freq == 0:
        checkin(i, losses)
    

vid_index = 0
def train(i):
    lossAll = ascend_txt()



user_guided.init(z)


# In[15]:


# 
torch.cuda.empty_cache()
import gc; gc.collect()
get_ipython().run_line_magic('pdb', 'off')

i = 0
import time
try:
    with tqdm() as pbar:
        while True and i != args.max_itter:
            if not webserver.us or not webserver.us.state.get("run"):
                pbar.set_description("Waiting for run")
                time.sleep(0.1)
                continue
            
            pbar.set_description("")
            
            train(i)

            if i > 0 and i%args.mse_decay_rate==0 and i <= args.mse_decay_rate * args.mse_epoches:
              
              opt = optim.Adam([z], lr=args.step_size, weight_decay=0.00000000)

            i += 1
            pbar.update()

except KeyboardInterrupt:
    pass


# # create video

# In[16]:


get_ipython().run_line_magic('cd', 'vids')

images = "%d.png"
video = "/content/my_video.mp4"
get_ipython().system('ffmpeg -r 30 -i $images -crf 20 -s 640x512 -pix_fmt yuv420p $video')
 
get_ipython().run_line_magic('cd', '..')

from google.colab import files
files.download('my_video.mp4')


# delete all frames from folder

# In[ ]:


get_ipython().run_line_magic('cd', 'vids')
get_ipython().run_line_magic('rm', '*.png')
get_ipython().run_line_magic('cd', '..')

