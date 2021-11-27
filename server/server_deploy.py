import os
import sys
import asyncio
import threading
from typing import *

import fastapi
import torchvision
import uvicorn
import numpy as np
from fastapi import WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from loguru import logger
from PIL import Image

from server.server_modelling import MaskOptimizer, ESRGAN
from server.server_modelling_utils import (
    process_mask,
    get_limits_from_mask,
    get_crop_tensor_from_img,
    scale_crop_tensor,
    merge_gen_img_into_canvas,
)
from server.server_data_utils import base64_to_pil, pil_to_base64
from server.server_config import DEBUG, DEBUG_OUT_DIR

app = fastapi.FastAPI()


class AsyncManager:
    """
    Manage updates in the async loop.
    """
    def __init__(self, ):
        """
        Set up async loop.
        """
        self.async_event_loop = asyncio.Event()
        self.async_value = None

    def set_async_value(
        self,
        async_value: Any,
    ) -> None:
        """
        Set async value and send event to the event loop.

        Args:
            async_value (Any): value to set on the async loop.

        """
        self.async_value = async_value
        self.async_event_loop.set()

    async def wait(self) -> None:
        """
        Waits until the event loop receives an event.

        """
        logger.info(f"AWAITING ASYNC DATA...")
        await self.async_event_loop.wait()

        logger.info(f"ASYNC DATA RECEIVED!")

        self.async_event_loop.clear()

        return self.async_value


class UserSession:
    """
    Functionalities and settings of each user connection.
    """
    def __init__(
        self,
        websocket: WebSocket,
    ) -> None:
        """
        Setup the setting of a user.

        Args:
            websocket (WebSocket): websocket used to communicate with the user.
        """
        self.websocket = websocket

        self.stop_generation = False
        self.mask_optimizer = None

        self.user_id = str(self.websocket['client'][1])

        self.async_manager = AsyncManager()

    async def run(self, ) -> None:
        """
        Launch listen and send async processes.
        """
        await asyncio.wait(
            [
                self.listen_loop(),
                self.send_loop(),
            ],
            return_when=asyncio.FIRST_COMPLETED,
        )

    def optimize_canvas(
        self,
        prompt: str,
        canvas_img: str,
        mask: str,
        lr: float = 0.5,
        style_prompt: str = "",
        padding_percent: float = 5.,
        num_rec_steps: int = 16,
        **kwargs,
    ) -> None:
        """
        Optimizes the region set by `mask` in the `canvas_img` using the settings provided.

        Args:
            prompt (str): prompt that will guide the optimization.
            canvas_img (str): base64 encoded canvas image to optimize.
            mask (str): base64 encoded mask determining the region to optimize in the canvas.
            lr (float, optional): learning rate. Defaults to 0.5.
            style_prompt (str, optional): prompt describing the style of the optimization. Defaults to "".
            padding_percent (float, optional): percent of external context to take into account in each generation. Defaults to 0.5.
            num_rec_steps (int, optional): Number of reconstruction steps. Defaults to 16.

        """
        canvas_img = base64_to_pil(canvas_img)
        canvas_img = np.float32(canvas_img.convert("RGB")) / 255.

        img_height, img_width, _ch = canvas_img.shape
        target_img_size = (
            img_width,
            img_height,
        )

        mask = base64_to_pil(mask)
        if DEBUG:
            os.makedirs(
                DEBUG_OUT_DIR,
                exist_ok=True,
            )
            mask.save(
                os.path.join(DEBUG_OUT_DIR,
                             f"{'_'.join(prompt.split())}_mask.png"))

        mask = process_mask(
            mask,
            target_img_size,
        )

        if DEBUG:
            os.makedirs(
                DEBUG_OUT_DIR,
                exist_ok=True,
            )
            Image.fromarray(np.uint8(mask * 255)).save(
                os.path.join(DEBUG_OUT_DIR,
                             f"{'_'.join(prompt.split())}_processed_mask.jpg"))

        crop_limits = get_limits_from_mask(
            mask,
            padding_percent,
        )

        img_crop_tensor = get_crop_tensor_from_img(
            canvas_img,
            crop_limits,
        )
        img_crop_tensor = scale_crop_tensor(img_crop_tensor)

        mask_crop_tensor = get_crop_tensor_from_img(
            mask[..., None],
            crop_limits,
        )
        mask_crop_tensor = scale_crop_tensor(mask_crop_tensor)

        if self.mask_optimizer is None:
            self.mask_optimizer = MaskOptimizer(
                prompt=prompt,
                cond_img=img_crop_tensor,
                mask=mask_crop_tensor,
                lr=lr,
                style_prompt=style_prompt,
            )

            self.mask_optimizer.optimize_reconstruction(
                num_iters=num_rec_steps, )

        gen_img = None
        optim_step = 0
        while not self.stop_generation:
            if self.stop_generation:
                break

            gen_img = self.mask_optimizer.optimize()

            updated_canvas = merge_gen_img_into_canvas(
                gen_img,
                mask_crop_tensor,
                canvas_img,
                crop_limits,
            )

            updated_canvas_pil = Image.fromarray(np.uint8(updated_canvas *
                                                          255))
            updated_canvas_uri = pil_to_base64(updated_canvas_pil)

            self.async_manager.set_async_value({
                "user_id": self.user_id,
                "image": updated_canvas_uri,
            })

            if DEBUG:
                os.makedirs(
                    DEBUG_OUT_DIR,
                    exist_ok=True,
                )

                gen_img_pil = torchvision.transforms.ToPILImage(mode="RGB")(
                    gen_img[0])
                gen_img_pil.save(
                    os.path.join(
                        DEBUG_OUT_DIR,
                        f"{'_'.join(prompt.split())}_{optim_step}.jpg"))

                updated_canvas_pil.save(
                    os.path.join(
                        DEBUG_OUT_DIR,
                        f"canvas_{'_'.join(prompt.split())}_{optim_step}.jpg"))

            optim_step += 1

        return

    def upscale_canvas(
        self,
        canvas_img: str,
        mask: str,
        padding_percent: int = 0,
        **kwargs,
    ) -> None:
        """
        Applies superresolution (using ESRGAN) to the area specified by the mask within the canvas and sends the result to the client using websockets.

        Args:
            canvas_img (str): canvas image where we want to apply superresolution.
            mask (str): mask of the canvas where superresolution is applied
            padding_percent (int, optional): percentage of padding used when computing the crop. Defaults to 0.
        """
        esrgan = ESRGAN()

        canvas_img = base64_to_pil(canvas_img)
        canvas_img = np.float32(canvas_img.convert("RGB")) / 255.

        img_height, img_width, _ch = canvas_img.shape
        target_img_size = (
            img_width,
            img_height,
        )

        mask = base64_to_pil(mask)
        mask = process_mask(
            mask,
            target_img_size,
        )

        crop_limits = get_limits_from_mask(
            mask,
            padding_percent,
        )

        img_crop_tensor = get_crop_tensor_from_img(
            canvas_img,
            crop_limits,
        )
        img_crop_tensor = scale_crop_tensor(img_crop_tensor)

        mask_crop_tensor = get_crop_tensor_from_img(
            mask[..., None],
            crop_limits,
        )
        mask_crop_tensor = scale_crop_tensor(mask_crop_tensor)

        _b, _ch, crop_height, crop_width = img_crop_tensor.shape

        num_chunks = int(np.ceil((crop_height * crop_width) / 256**2))
        upscaled_crop = esrgan.upscale_img(
            img_crop_tensor,
            num_chunks,
        )

        updated_canvas = merge_gen_img_into_canvas(
            upscaled_crop,
            mask_crop_tensor,
            canvas_img,
            crop_limits,
        )

        updated_canvas_pil = Image.fromarray(np.uint8(updated_canvas * 255))
        updated_canvas_uri = pil_to_base64(updated_canvas_pil)

        self.async_manager.set_async_value({
            "user_id": self.user_id,
            "image": updated_canvas_uri,
        })

    async def listen_loop(self, ):
        """
        Handle incoming messages from the client.
        """
        try:
            while True:
                msg_dict = await self.websocket.receive_json()

                topic = msg_dict["topic"]
                logger.info(f"RECEIVED TOPIC {topic}")

                data_dict = msg_dict["data"]

                if topic == "initialize":
                    self.initialize = True

                if topic == "upscale-generation":
                    print(data_dict)
                    optimize_kwargs = {
                        "canvas_img": data_dict["backgroundImg"],
                        "mask": data_dict["imageBase64"],
                    }

                    upscale_thread = threading.Thread(
                        target=self.upscale_canvas,
                        kwargs=optimize_kwargs,
                    )
                    upscale_thread.start()

                elif topic == "start-generation" or topic == "resume-generation":
                    if topic != "resume-generation":
                        self.mask_optimizer = None

                    self.stop_generation = False

                    optimize_kwargs = {
                        "prompt": data_dict["prompt"],
                        "canvas_img": data_dict["backgroundImg"],
                        "mask": data_dict["imageBase64"],
                        "lr": data_dict["learningRate"],
                        "style_prompt": data_dict["stylePrompt"],
                        "padding_percent": 10.,
                        "num_rec_steps": data_dict["numRecSteps"],
                    }

                    optimize_layer_thread = threading.Thread(
                        target=self.optimize_canvas,
                        kwargs=optimize_kwargs,
                    )
                    optimize_layer_thread.start()

                elif topic == "stop-generation":
                    self.stop_generation = True

        except Exception as e:
            logger.exception("Error", e)

    async def send_loop(self):
        """
        Handle the emission of messages to the client.
        """
        while True:
            result_dict = await self.async_manager.wait()
            if self.user_id == result_dict["user_id"]:
                await self.websocket.send_json(result_dict)

            logger.info(f"{self.user_id} ASYNC RESULTS SENT!")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, ) -> None:
    """
    Handle websocket connections

    Args:
        websocket (WebSocket): incoming websocket connection.
    """
    try:
        await websocket.accept()
        logger.info(f"WEBSOCKET CONNECTED!")

        user_session = UserSession(websocket, )

        await user_session.run()

    except Exception as e:
        logger.error(f"WEBSOCKET CONNECTION ERROR: {e}")

    finally:
        logger.info("WEBSOCKET DISCONNECTED.")

    return

@app.get("/health")
async def index():
    return "ok"

STATIC_FOLDERS = [
    "images", # public images from public folder
    "assets"  # Built assets from vite
]

# Add static path if configured
if os.environ.get("STATIC_PATH"):
    static_root = os.environ["STATIC_PATH"]
    for static_folder in STATIC_FOLDERS:
        static_path = f"{static_root}/{static_folder}"
        app.mount(f"/{static_folder}", StaticFiles(directory=static_path), name=static_folder)
    
    @app.get("/")
    async def index():
        return FileResponse(f"{static_root}/index.html")


def main():
    """
    Launch the app.
    """
    logger.info("Starting server...")

    loop = "asyncio"
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8004)),
        loop=loop,
    )

    return


if DEBUG:
    level = "DEBUG"
else:
    level = "DEBUG"

logger.remove()
logger.add(
    sys.stderr,
    level=level,
)

if __name__ == "__main__":
    main()