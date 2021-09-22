
import json
from typing import *
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import io
import base64
import torch
import asyncio
import fastapi
import uvicorn
import numpy as np
from fastapi import WebSocket
from loguru import logger
from torchvision.transforms import functional as TF

app = fastapi.FastAPI()

polygon_worker = ThreadPoolExecutor(1)
inferece_worker = ThreadPoolExecutor(1)


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


class UserSession:
    def __init__(
        self,
        websocket: WebSocket,
    ):
        self.websocket = websocket
        self.coord = None
        self.prompt = "3D bunny rabbit gray mesh rendered with maya zbrush"
        self.run_tick = True
        self.initialize = None
        self.state = {}

    async def run(self):
        await asyncio.wait([self.listen_loop(), self.send_loop()], return_when=asyncio.FIRST_COMPLETED)
        self.run_tick = False  # stop running optimization if we die

    async def listen_loop(self):
        try:
            while True:
                cmd = await self.websocket.receive_text()
                cmd = json.loads(cmd)
                
                topic = cmd['topic']
                data = cmd['data']


                print("XXX Got cmd", cmd)
                if topic == 'initialize':
                    self.initialize = True

                elif topic == 'state':
                    self.state = data
                
                elif topic == "sculp_settings":
                    if data:
                        self.run_tick = data["sculp_enabled"]
                        self.prompt = data["prompt"]
        except:
            logger.exception("Error")

    async def send_loop(self):
        while True:
            model = await async_result.wait()
            print(f"XXX WS sending model, {len(model)//1024}kb")
            await self.websocket.send_text(model)



us: Optional[UserSession] = None
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global us
    await websocket.accept()
    new_session = UserSession(websocket)
    us = new_session
    
    # optimization_worker.user_session = us
    await new_session.run()


def process_step(output: Union[np.ndarray, torch.Tensor], loss_dict: Dict[str, float]={}):
    print("loss", " ".join(f"{k}: {v:.4f}" for k, v in loss_dict.items()))
    start = datetime.now()

    logger.debug("to PIL")
    x = TF.to_pil_image(output[0])
    buffer = io.BytesIO()
    x.save(buffer, "jpeg")

    logger.debug("XXX dur")
    encoded = base64.standard_b64encode(buffer.getvalue()).decode()
    result = dict(image=encoded, **loss_dict)

    if async_result:
        async_result.set(json.dumps(result))
        logger.info("XXX Posted message")
    else:
        logger.info("XXX NO NOTIFIER???")
        print("not Posted")

    dur = datetime.now() - start
    print("Took", dur.total_seconds())


#%%
def on_update(*args, **kwargs):
    logger.info("XXX enqueuing preprocess")
    polygon_worker.submit(process_step, *args, **kwargs)




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