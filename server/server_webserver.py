import sys
import json
import io
import base64
import asyncio
import logging
import threading
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
from PIL import Image
from torchvision.transforms import functional as TF

from server.server_data_utils import (
    process_mask,
    get_limits_from_mask,
    get_crop_from_limits,
    scale_crop,
    merge_gen_img_into_canvas,
)

sys.path.append("../")
sys.path.append("../HuggingGAN")
from server.server_model_utils import LayerOptimizer

logging.basicConfig(
    format="%(levelname)s:%(message)s",
    level=logging.DEBUG,
)

app = fastapi.FastAPI()

update_worker = ThreadPoolExecutor(1)


class AsyncResult:
    def __init__(self, ):
        self.async_event_loop = asyncio.Event()
        self.async_value = None

    def set_async_value(
        self,
        async_value,
    ):
        self.async_value, = async_value
        self.async_event_loop.set()

    async def wait(self):
        logging.info(f"AWAITING ASYNC DATA...")
        await self.async_event_loop.wait()

        logging.info(f"ASYNC DATA RECEIVED!")

        self.signal.clear()

        return self.async_value


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


def decode_base64_img(base64_img, ):
    base64_img = base64_img.split(",")[-1]
    img = base64.b64decode(base64_img)
    img = io.BytesIO(img)
    img = Image.open(img)
    img = np.array(img)

    return img


class UserSession:
    def __init__(
        self,
        websocket: WebSocket,
    ):
        self.websocket = websocket

        self.user_id = str(self.websocket['client'][1])

        self.layer_optimizer = LayerOptimizer()
        self.stop_optimization = False

    async def run(self):
        await asyncio.wait(
            [
                self.listen_loop(),
                self.send_loop(),
            ],
            return_when=asyncio.FIRST_COMPLETED,
        )

    def optimize_layer(
        self,
        prompt: str,
        canvas_img: str,
        mask: str,
        lr: float = 0.05,
        num_optim_steps: int = 16,
        style_prompt: str = "",
        **kwargs,
    ):
        canvas_img = decode_base64_img(canvas_img)
        canvas_img = np.float32(canvas_img.convert("RGB")) / 255.

        img_height, img_width, _ch = canvas_img.shape
        target_img_size = (
            img_width,
            img_height,
        )

        mask = decode_base64_img(mask)
        mask = process_mask(
            mask,
            target_img_size,
        )

        crop_limits = get_limits_from_mask(mask, )

        img_crop = get_crop_from_limits(
            canvas_img,
            crop_limits,
        )
        img_crop = scale_crop(img_crop)

        mask_crop = get_crop_from_limits(
            mask[..., None],
            crop_limits,
        )
        mask_crop = scale_crop(mask_crop)

        layer = LayerOptimizer(
            prompt=prompt,
            cond_img=img_crop,
            mask=mask_crop,
            lr=lr,
        )

        gen_img = None
        for optim_step in range(num_optim_steps, ):
            gen_img = layer.optimize_layer()

            # gen_img_pil = torchvision.transforms.ToPILImage(mode="RGB")(
            #     gen_img[0])
            # gen_img_pil.save(
            #     f"generations/{'_'.join(prompt.split())}_{optim_step}.jpg")

            if self.stop_layer_optim:
                break

        updated_canvas = merge_gen_img_into_canvas(
            gen_img,
            mask_crop,
            canvas_img,
            crop_limits,
        )

        # Image.fromarray(np.uint8(updated_canvas * 255)).save(
        #     f"generations/final_{'_'.join(prompt.split())}_{optim_step}.jpg")

        return updated_canvas

    async def listen_loop(self, ):
        try:
            while True:
                data_dict = await self.websocket.receive_json()

                topic = data_dict["topic"]
                logging.info(f"GOT TOPIC {topic}")

                data_list = data_dict["data"]

                if topic == "initialize":
                    self.initialize = True

                elif topic == "state":
                    self.state = data_list
                    self.stop_generation = True

                elif topic == "start-generation":
                    self.stop_generation = False
                    self.optimize_layer(data_list)

                    self.stop_generation = False

                elif topic == "stop-generation":
                    self.stop_generation = True

        except:
            logger.exception("Error")

    async def send_loop(self):
        while True:
            results = await async_result.wait()
            await self.websocket.send_json(results)
            print(f"{self.user_id} ASYNC RESULTS SENT!")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, ):
    try:
        await websocket.accept()
        logging.info(f"WEBSOCKET CONNECTED!")

        user_session = UserSession(websocket)

        await user_session.run()

    except Exception as e:
        logging.info(f"WEBSOCKET CONNECTION ERROR: {e}")

    finally:
        logging.info("WEBSOCKET DISCONNECTED.")


def process_step(
    output: Union[np.ndarray, torch.Tensor],
    loss_dict: Dict[str, float] = {},
):

    logger.debug("loss",
                 " ".join(f"{k}: {v:.4f}" for k, v in loss_dict.items()))

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
        async_result.set_async_value(json.dumps(result))
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