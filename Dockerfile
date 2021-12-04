FROM node:16-alpine as builder

WORKDIR /client

COPY client/package.json client/package-lock.json ./

RUN npm install esbuild
RUN npm install

COPY client ./

RUN npm run build

# Use nvidia/cuda image
# https://stackoverflow.com/questions/65492490/how-to-install-cuda-enabled-pytorch-in-a-docker-container
FROM nvidia/cuda:11.0-cudnn8-runtime-ubuntu18.04

# set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# Install anaconda
RUN apt-get update \
  && apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion \
  && apt-get clean

RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh -O ~/anaconda.sh \
  && /bin/bash ~/anaconda.sh -b -p /opt/conda \
  && rm ~/anaconda.sh \
  && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
  && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
  && find /opt/conda/ -follow -type f -name '*.a' -delete \
  && find /opt/conda/ -follow -type f -name '*.js.map' -delete \
  && /opt/conda/bin/conda clean -afy

# Set path to conda
ENV PATH /opt/conda/bin:$PATH

WORKDIR /server

COPY server/env-server.yml ./

# Install scipy deps
RUN apt-get install -y gcc-8 g++-8

RUN conda update conda \
  && conda env create -q -f ./env-server.yml \
  # Steps to reduce docker image size from https://jcristharif.com/conda-docker-tips.html
  && conda clean -afy \
  && find /opt/conda/ -follow -type f -name '*.a' -delete \
  && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
  && find /opt/conda/ -follow -type f -name '*.js.map' -delete

RUN echo "conda activate prosepaint" >> ~/.bashrc
ENV PATH /opt/conda/envs/prosepaint/bin:$PATH
ENV CONDA_DEFAULT_ENV prosepaint

ENV MODELING_DIR=/opt/conda/envs/prosepaint/lib/python3.7/site-packages/geniverse/models

# Pre-download model weights - see https://github.com/thegeniverse/geniverse/blob/main/geniverse/models/taming/modeling_taming.py
# and make sure that model URLs and filenames stay lined up.
RUN mkdir -p $MODELING_DIR/taming/.modeling_cache \
  && wget -q -O $MODELING_DIR/taming/.modeling_cache/imagenet-16384.ckpt https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1 \
  && wget -q -O $MODELING_DIR/taming/.modeling_cache/imagenet-16384.yaml https://raw.githubusercontent.com/vipermu/taming-transformers/master/configs/imagenet-16384.yaml \
  && mkdir -p /root/.cache/torch/hub/checkpoints/ \
  && wget -q -O /root/.cache/torch/hub/checkpoints/vgg16-397923af.pth https://download.pytorch.org/models/vgg16-397923af.pth \
  # CLIP from https://github.com/thegeniverse/geniverse/blob/main/geniverse/modeling_utils.py
  && mkdir -p /root/.cache/clip \
  && wget -q -O /root/.cache/clip/ViT-B-32.pt https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt \
  # VGG LPIPS from https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/lpips.py
  && wget -q -O $MODELING_DIR/taming/.modeling_cache/vgg.pth https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1

COPY server ./

COPY --from=builder /client/dist /client-dist

ENV PYTHONPATH=/
ENV PORT=80
ENV STATIC_PATH=/client-dist

CMD ["conda", "run", "--no-capture-output", "-n", "prosepaint", "python", "/server/server_deploy.py"]

EXPOSE 80