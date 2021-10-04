import json
import io
import base64
import asyncio
from typing import *
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch
import fastapi
import uvicorn
import numpy as np
from fastapi import WebSocket
from loguru import logger
from torchvision.transforms import functional as TF
from PIL import Image

from vqgan import LayeredGenerator

app = fastapi.FastAPI()

update_worker = ThreadPoolExecutor(1)


class AsyncResult:
    def __init__(self):
        loop = asyncio.get_event_loop()
        self.signal = asyncio.Event(loop=loop)
        self.value = None

    def set(self, value):
        self.value = value
        self.signal.set()

    async def wait(self):
        await self.signal.wait()
        self.signal.clear()
        return self.value


async_result = None


@app.on_event("startup")
async def startup_event():
    global async_result
    async_result = AsyncResult()


@dataclass
class Layer:
    color: str
    strength: int
    prompt: str
    img: np.ndarray


def decode_layer(data_dict, ):
    # decode image
    x = data_dict["imageBase64"]
    if not x:
        return None

    x = x.split(",")[-1]
    x = base64.b64decode(x)
    x = io.BytesIO(x)
    x = Image.open(x)
    x = np.array(x)

    return Layer(
        color=data_dict["color"],
        strength=data_dict["strength"],
        prompt=data_dict["prompt"],
        img=x,
    )


class UserSession:
    def __init__(
        self,
        websocket: WebSocket,
    ):
        self.websocket = websocket
        self.coord = None
        self.run_tick = True
        self.initialize = None
        self.state = {}
        self.layer_list = []

        self.stop_generation = False
        self.max_gen_iterations = 200

        self.layered_generator = LayeredGenerator(layer_list=self.layer_list)

    async def run(self):
        await asyncio.wait(
            [
                self.listen_loop(),
                self.send_loop(),
            ],
            return_when=asyncio.FIRST_COMPLETED,
        )
        self.run_tick = False  # stop running optimization if we die

    async def listen_loop(self):
        try:
            while True:
                cmd = await self.websocket.receive_text()
                cmd = json.loads(cmd)

                topic = cmd["topic"]
                data_list = cmd["data"]

                print("XXX Got cmd", topic)
                if topic == "initialize":
                    self.initialize = True

                elif topic == "state":
                    self.state = data_list

                elif topic == "start-generation":
                    self.layer_list = [
                        decode_layer(data_dict) for data_dict in data_list
                    ]
                    self.layer_list = [
                        layer for layer in self.layer_list if layer
                    ]

                    self.layered_generator.reset_layers(self.layer_list)

                    counter = 0
                    while not self.stop_generation or counter > self.max_gen_iterations:
                        counter += 1
                        output, loss_dict, state = self.layered_generator()
                        process_step(output, loss_dict)

                    self.stop_generation = False

                elif topic == "stop-generation":
                    self.stop_generation = True

        except:
            logger.exception("Error")

    async def send_loop(self):
        while True:
            results = await async_result.wait()
            print(f"XXX WS sending results, {results}")
            await self.websocket.send_text(results)


us: Optional[UserSession] = None


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, ):
    global us
    await websocket.accept()
    new_session = UserSession(websocket)
    us = new_session

    print("XXX: user session created")

    # optimization_worker.user_session = us
    await new_session.run()


def process_step(
    output: Union[np.ndarray, torch.Tensor],
    loss_dict: Dict[str, float] = {},
):

    print("loss", " ".join(f"{k}: {v:.4f}" for k, v in loss_dict.items()))
    loss_dict = {k: float(v.detach().cpu()) for k, v in loss_dict.items()}
    start = datetime.now()

    logger.debug("to PIL")
    x = TF.to_pil_image(output[0])
    buffer = io.BytesIO()
    x.save(buffer, "jpeg")

    logger.debug("XXX dur")
    encoded = base64.standard_b64encode(buffer.getvalue()).decode()
    result = dict(
        image=encoded,
        **loss_dict,
    )

    if async_result:
        async_result.set(json.dumps(result))
        logger.info("XXX Posted message")
    else:
        logger.info("XXX NO NOTIFIER???")
        print("not Posted")

    dur = datetime.now() - start
    print("Took", dur.total_seconds())


def on_update(*args, **kwargs):
    logger.info("XXX enqueuing preprocess")
    update_worker.submit(process_step, *args, **kwargs)


def main():
    loop = "asyncio"
    print("Starting server...")
    # thread = threading.Thread(
    #     target=run_sdf_clip,
    #     daemon=True,
    # )
    # thread.start()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8005,
        loop=loop,
    )


import sys

logger.remove()
logger.add(sys.stderr, level="INFO")

if __name__ == "__main__":
    main()