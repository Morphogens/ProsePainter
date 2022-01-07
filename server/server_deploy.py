import os
import sys
import asyncio
import threading
from typing import *

import fastapi
import uvicorn
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from loguru import logger
from PIL import Image
from upscaler.models import ESRGAN, ESRGANConfig

from server.server_modeling_utils import (
    process_mask,
    get_limits_from_mask,
    get_crop_tensor_from_img,
    scale_crop_tensor,
    merge_gen_img_into_canvas,
)
from server.server_data_utils import base64_to_pil, pil_to_base64
from server.server_config import DEBUG, DEBUG_OUT_DIR
from server.server_queue_utils import OptimizationManager
from server.server_async import AsyncManager

app = fastapi.FastAPI()

async_manager = None
async_manager_running = False
optimization_manager = None


class UserSession:
    """
    Functionalities and settings of each user connection.
    """
    def __init__(
        self,
        user_id,
        websocket: WebSocket,
    ) -> None:
        """
        Setup the setting of a user.

        Args:
            websocket (WebSocket): websocket used to communicate with the user.
        """
        self.websocket = websocket

        self.mask_optimizer = None

        self.user_id = user_id

    def optimize_canvas(
        self,
        prompt: str,
        canvas_img: str,
        mask: str,
        lr: float = 0.5,
        style_prompt: str = "",
        padding_percent: float = 10.,
        num_rec_steps: int = 16,
        model_type: str = "imagenet-16384",
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
        # if DEBUG:
        #     os.makedirs(
        #         DEBUG_OUT_DIR,
        #         exist_ok=True,
        #     )
        #     mask.save(
        #         os.path.join(DEBUG_OUT_DIR,
        #                      f"{'_'.join(prompt.split())}_mask.png"))

        mask = process_mask(
            mask,
            target_img_size,
        )

        # if DEBUG:
        #     os.makedirs(
        #         DEBUG_OUT_DIR,
        #         exist_ok=True,
        #     )
        #     Image.fromarray(np.uint8(mask * 255)).save(
        #         os.path.join(DEBUG_OUT_DIR,
        #                      f"{'_'.join(prompt.split())}_processed_mask.jpg"))

        crop_limits = get_limits_from_mask(
            mask,
            padding_percent,
        )

        if crop_limits is None:
            logger.error("No mask!")

            async_manager.set_async_value(
                user_id=self.user_id,
                async_value={
                    "error": "no mask!",
                },
                websocket=self.websocket,
            )

            return

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

        prompt = f"{prompt} {style_prompt}"

        optimization_manager.add_job(
            user_id=self.user_id,
            prompt=prompt,
            cond_img=img_crop_tensor,
            mask=mask_crop_tensor,
            mask_crop_tensor=mask_crop_tensor,
            canvas_img=canvas_img,
            crop_limits=crop_limits,
            websocket=self.websocket,
        )

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

        esrgan_config = ESRGANConfig()
        esrgan = ESRGAN(esrgan_config)

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

        upscaled_crop = esrgan.upscale(img_crop_tensor)

        updated_canvas = merge_gen_img_into_canvas(
            upscaled_crop,
            mask_crop_tensor,
            canvas_img,
            crop_limits,
        )

        updated_canvas_pil = Image.fromarray(np.uint8(updated_canvas * 255))
        updated_canvas_uri = pil_to_base64(updated_canvas_pil)

        # self.async_manager.set_async_value({
        #     "user_id": self.user_id,
        #     "image": updated_canvas_uri,
        # })

    async def listen_loop(self, ):
        """
        Handle incoming messages from the client.
        """
        try:
            async_manager.add_user(
                self.user_id,
                self.websocket,
            )

            while True:
                try:
                    msg_dict = await self.websocket.receive_json()
                except WebSocketDisconnect:
                    break

                topic = msg_dict["topic"]
                logger.info(f"RECEIVED TOPIC {topic}")

                data_dict = msg_dict["data"]

                if topic == "start-generation" or topic == "resume-generation":
                    if topic != "resume-generation":
                        self.mask_optimizer = None

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
                    optimization_manager.remove_job(self.user_id, )

                # elif topic == "upscale-generation":
                #     print(data_dict)
                #     optimize_kwargs = {
                #         "canvas_img": data_dict["backgroundImg"],
                #         "mask": data_dict["imageBase64"],
                #     }

                #     upscale_thread = threading.Thread(
                #         target=self.upscale_canvas,
                #         kwargs=optimize_kwargs,
                #     )
                #     upscale_thread.start()

        except Exception as e:
            logger.exception("Error", e)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, ) -> None:
    """
    Handle websocket connections

    Args:
        websocket (WebSocket): incoming websocket connection.
    """
    user_id = str(websocket['client'][1])

    run_send_loop = False

    try:
        await websocket.accept()
        logger.info(f"WEBSOCKET CONNECTED!")

        global async_manager
        if async_manager is None:
            logger.info("SETTING UP ASYNC MANAGER...")
            async_manager = AsyncManager()
            run_send_loop = True

        global optimization_manager
        if optimization_manager is None:
            logger.info("SETTING UP OPTIMIZATION MANAGER...")
            optimization_manager = OptimizationManager(async_manager, )
            optimization_manager.start()

        user_session = UserSession(
            user_id,
            websocket,
        )

        if run_send_loop:
            co_routine_list = [
                async_manager.send_loop(),
                user_session.listen_loop(),
            ]
        else:
            co_routine_list = [
                user_session.listen_loop(),
            ]

        await asyncio.wait(
            co_routine_list,
            return_when=asyncio.FIRST_COMPLETED,
        )

    except Exception as e:
        logger.error(f"WEBSOCKET CONNECTION ERROR: {e}")

    finally:
        logger.info("WEBSOCKET DISCONNECTED.")
        async_manager.remove_user(user_id, )
        optimization_manager.remove_job(user_id, )

    return


@app.get("/health")
async def index():
    return "ok"


STATIC_FOLDERS = [
    "images",  # public images from public folder
    "assets"  # Built assets from vite
]

# Add static path if configured
if os.environ.get("STATIC_PATH"):
    static_root = os.environ["STATIC_PATH"]
    for static_folder in STATIC_FOLDERS:
        static_path = f"{static_root}/{static_folder}"
        app.mount(f"/{static_folder}",
                  StaticFiles(directory=static_path),
                  name=static_folder)

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
logger.disable("uvicorn")
logger.add(
    sys.stderr,
    level=level,
)

if __name__ == "__main__":
    main()
