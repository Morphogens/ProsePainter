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

COPY server ./

COPY --from=builder /client/dist /client-dist

ENV PYTHONPATH=/
ENV PORT=80
ENV STATIC_PATH=/client-dist

CMD ["conda", "run", "--no-capture-output", "-n", "prosepaint", "python", "/server/server_deploy.py"]

EXPOSE 80