# ProsePainter 
Create images by ***painting with words***.

ProsePainter combines direct digital painting with real-time guided machine-learning based image optimization. Simply state what you want and draw the region where you want it. 
![Tux, the Linux mascot](preview.jpg)
 
## Setup
The app consists of a python server which does the computations and a web based interface using a nodejs server.

### Install python server
Due to the use of CLIP, it is recommended to use **python3.7** and **torch 1.7.1+cu110** (available [here](https://pytorch.org/get-started/previous-versions/)).

With the following command you can set up a conda environment named _prosepaint_ where all the dependencies will be installed.
```bash
conda env create -f server/env-server.yml
```

### Run python server
The following command will launch the uvicorn server.
```bash
export PYTHONPATH=.; python server/server_deploy.py
```

### Install node server
```
cd client
npm install
```
### Run node server
```
cd client
npm run dev
```
Open http://localhost:8003/ in your web browser.

### Build docker container

```
docker build . -t prosepainter:latest
```

### Run python server with docker

```
docker run -p 80:8004 prosepainter:latest
```