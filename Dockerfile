FROM node:16-alpine as builder

WORKDIR /client

COPY client/package.json client/package-lock.json ./

RUN npm install esbuild
RUN npm install

COPY client ./

RUN npm run build

# https://github.com/aws/deep-learning-containers/blob/master/available_images.md
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.9.1-gpu-py38-cu111-ubuntu20.04

WORKDIR /server

COPY server/env-server.yml ./

RUN conda env create -f ./env-server.yml

COPY --from=builder /client/dist /client-dist

ENV PYTHONPATH=/server
ENV PORT 80

CMD ["conda", "run", "--no-capture-output", "-n", "./env-server.yml", "python", "./server_deploy.py"]

EXPOSE 80