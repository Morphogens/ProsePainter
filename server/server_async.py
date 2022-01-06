import asyncio
import time
from typing import *

from collections import defaultdict
from loguru import logger
from fastapi import WebSocket


class AsyncManager:
    def __init__(self, ):
        self.async_event_loop = asyncio.Event()
        self.async_value_buffer = defaultdict(lambda: [], )

    async def wait_for_async_result(self, ):
        await self.async_event_loop.wait()
        self.async_event_loop.clear()

        return

    def set_async_value(
        self,
        user_id: str,
        async_value: Dict,
        websocket: WebSocket,
    ):
        if not isinstance(async_value, dict):
            logger.warning(f"{user_id} AYNC VALUE MUST BE A DICT!")
            return

        async_value["websocket"] = websocket

        self.async_value_buffer[user_id].append(async_value)
        self.async_event_loop.set()

        logger.debug(f"{user_id} ASYNC VALUE ADDED")

        time.sleep(0)

        return

    async def send_async_data(self, ):
        for user_id in self.async_value_buffer.keys():
            async_value_list = self.async_value_buffer.pop(user_id)

            for _async_idx in range(len(async_value_list)):
                async_value = async_value_list.pop()

                websocket = async_value.pop("websocket")
                await websocket.send_json(async_value, )

                logger.info(f"{user_id} ASYNC RESULTS SENT!")

                await asyncio.sleep(0.)

            return

    async def send_loop(self):
        """
        Handle the emission of messages to the client.
        """
        while True:
            print("WAITING FOR ASYNC RESULTS")
            await self.wait_for_async_result()
            await self.send_async_data()