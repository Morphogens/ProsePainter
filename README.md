## Dependencies
```sh
$ git submodule update --init --recursive

$ cd server
$ pip install -r requirements.txt
$ curl -L 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1' > vqgan_imagenet_f16_16384.yaml
$ curl -L 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1' > vqgan_imagenet_f16_16384.ckpt
```


## Run it
```sh
$ cd server && python np_app.py
```

## Switching to other notebooks
The patches are:

call `webserver.init(z)`

call `webserver.process_step(tensor)` on each step to push result to the client

Then at the optimization loop, right before backwards is called, add:

```python
if webserver.us:
    user_guided.update_state(webserver.us.state)
    loss += user_guided.apply_losses(iii)
else:
    user_guided.update({})
```

on the outer loop, add:
```py
if not webserver.us or not webserver.us.state.get("run"):
    time.sleep(0.1)
    continue
```

Currently the loss it returns is 0, and the latent is updated internally.
This will be refactored Soon™