## SERVER SET UP
### INSTALL DEPENDENCIES
Due to the use of CLIP, it is recommended to use **python3.7** and **torch 1.7.1+cu110** (available [here](https://pytorch.org/get-started/previous-versions/)).

With the following command you can set up a conda environment named _nuclear_ where all the dependencies will be installed.
```bash
conda env create -f server/env-server.yaml
```

## LAUNCH THE SERVER
The following command will launch the uvicorn server.
```bash
export PYTHONPATH=.; python server/server_deploy.py

```