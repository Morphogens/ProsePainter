import os
import sys
import asyncio
import threading
from typing import *

import fastapi
import uvicorn
import numpy as np
from fastapi import WebSocket
from loguru import logger
from PIL import Image

from server.server_modelling import LayerOptimizer
from server.server_modelling_utils import (
    process_mask,
    get_limits_from_mask,
    get_crop_from_limits,
    scale_crop,
    merge_gen_img_into_canvas,
)
from server.server_data_utils import decode_base64_img, pil_to_base64
from server.server_config import LOGGER_LEVEL

app = fastapi.FastAPI()


class AsyncResult:
    def __init__(self, ):
        self.async_event_loop = asyncio.Event()
        self.async_value = None

    def set_async_value(
        self,
        async_value,
    ):
        self.async_value = async_value
        self.async_event_loop.set()

    async def wait(self):
        logger.info(f"AWAITING ASYNC DATA...")
        await self.async_event_loop.wait()

        logger.info(f"ASYNC DATA RECEIVED!")

        self.async_event_loop.clear()

        return self.async_value


async_result = None


@app.on_event("startup")
async def startup_event():
    global async_result
    async_result = AsyncResult()


class UserSession:
    def __init__(
        self,
        websocket: WebSocket,
    ):
        self.websocket = websocket

        self.stop_generation = False

        self.user_id = str(self.websocket['client'][1])

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
        lr: float = 0.5,
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
        mask.save(f"generations/{'_'.join(prompt.split())}_mask.png")
        mask = process_mask(
            mask,
            target_img_size,
        )
        Image.fromarray(np.uint8(mask * 255)).save(
            f"generations/{'_'.join(prompt.split())}_processed_mask.png")

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
        counter = 0
        while not self.stop_generation:
            gen_img = layer.optimize()

            # gen_img_pil = torchvision.transforms.ToPILImage(mode="RGB")(
            #     gen_img[0])
            # gen_img_pil.save(
            #     f"generations/{'_'.join(prompt.split())}_{optim_step}.jpg")

            updated_canvas = merge_gen_img_into_canvas(
                gen_img,
                mask_crop,
                canvas_img,
                crop_limits,
            )

            updated_canvas_pil = Image.fromarray(np.uint8(updated_canvas *
                                                          255))
            updated_canvas_uri = pil_to_base64(updated_canvas_pil)

            async_result.set_async_value({
                "image": updated_canvas_uri,
            })

            # os.makedirs(
            #     "generations",
            #     exist_ok=True,
            # )
            # updated_canvas_pil.save(
            #     f"generations/canvas_{'_'.join(prompt.split())}_{counter}.jpg")

            if self.stop_generation:
                break

            counter += 1

        # Image.fromarray(np.uint8(updated_canvas * 255)).save(
        #     f"generations/final_{'_'.join(prompt.split())}_{optim_step}.jpg")

        return updated_canvas

    async def listen_loop(self, ):
        try:
            while True:
                msg_dict = await self.websocket.receive_json()

                topic = msg_dict["topic"]
                logger.info(f"GOT TOPIC {topic}")

                data_dict = msg_dict["data"]

                if topic == "initialize":
                    self.initialize = True

                elif topic == "start-generation":
                    self.stop_generation = False

                    prompt = data_dict["prompt"]
                    canvas_img = data_dict["backgroundImg"]
                    mask = data_dict["imageBase64"]
                    lr = data_dict["learningRate"]

                    optimize_layer_thread = threading.Thread(
                        target=self.optimize_layer,
                        args=(
                            prompt,
                            canvas_img,
                            mask,
                            lr,
                        ),
                    )
                    optimize_layer_thread.start()

                elif topic == "stop-generation":
                    self.stop_generation = True

        except Exception as e:
            logger.exception("Error", e)

    async def send_loop(self):
        while True:
            result_dict = await async_result.wait()
            await self.websocket.send_json(result_dict)

            logger.info(f"{self.user_id} ASYNC RESULTS SENT!")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, ):
    try:
        await websocket.accept()
        logger.info(f"WEBSOCKET CONNECTED!")

        user_session = UserSession(websocket, )

        await user_session.run()

    except Exception as e:
        logger.info(f"WEBSOCKET CONNECTION ERROR: {e}")

    finally:
        logger.info("WEBSOCKET DISCONNECTED.")


def main():
    logger.info("Starting server...")

    loop = "asyncio"
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8004,
        loop=loop,
    )


logger.remove()
logger.add(
    sys.stderr,
    level=LOGGER_LEVEL,
)

if __name__ == "__main__":
    main()