FROM node:16-alpine as builder

WORKDIR /client

COPY client/package.json client/package-lock.json ./

RUN npm install esbuild
RUN npm install

COPY client ./

RUN npm run build

FROM pytorch/pytorch

WORKDIR /server

COPY server/env-server.yml ./

RUN conda env create -f ./env-server.yml

COPY --from=builder /client/dist /client-dist

ENV PYTHONPATH=/server
ENV PORT 80

CMD ["conda", "run", "--no-capture-output", "-n", "./env-server.yml", "python", "./server_deploy.py"]

EXPOSE 80