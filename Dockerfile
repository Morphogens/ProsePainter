FROM node:16-alpine as builder

WORKDIR /client

COPY client/package.json client/package-lock.json ./

RUN npm install esbuild
RUN npm install

COPY client ./

RUN npm run build

FROM continuumio/miniconda3

# Fix for https://github.com/lhelontra/tensorflow-on-arm/issues/13
RUN apt-get -qq update \
  && apt-get install -qqy gpg software-properties-common \
  && apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 1E9377A2BA9EF27F \
  && add-apt-repository ppa:ubuntu-toolchain-r/test \
  && apt-get -qq update \
  && apt-get -qqy upgrade

WORKDIR /server

COPY server/env-server.yml ./

RUN conda env create -q -f ./env-server.yml \
  # Steps to reduce docker image size from https://jcristharif.com/conda-docker-tips.html
  && conda clean -afy \
  && find /opt/conda/ -follow -type f -name '*.a' -delete \
  && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
  && find /opt/conda/ -follow -type f -name '*.js.map' -delete

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
  && wget -q -O https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1 $MODELING_DIR/taming/.modeling_cache/vgg.pth

COPY server ./

COPY --from=builder /client/dist /client-dist

ENV PYTHONPATH=/
ENV PORT=80
ENV STATIC_PATH=/client-dist

CMD ["conda", "run", "--no-capture-output", "-n", "prosepaint", "python", "/server/server_deploy.py"]

EXPOSE 80