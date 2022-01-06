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

        self.num_users = 0
        self.active_user_list = []
        self.websocket_list = []

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
        user_id_list = list(self.async_value_buffer.keys())
        for user_id in user_id_list:
            async_value_list = self.async_value_buffer.pop(user_id)[::-1]

            if user_id not in self.active_user_list:
                continue

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

    def add_user(
        self,
        user_id,
        websocket,
    ):
        self.num_users += 1
        self.active_user_list.append(user_id, )
        self.websocket_list.append(websocket, )

        for u_id, ws in zip(self.active_user_list, self.websocket_list):
            self.set_async_value(
                u_id,
                {
                    'message': 'userCount',
                    'numUsers': self.num_users,
                },
                ws,
            )

    def remove_user(
        self,
        user_id,
    ):
        if user_id not in self.active_user_list:
            return

        self.num_users -= 1

        user_idx = self.active_user_list.index(user_id)
        self.active_user_list.pop(user_idx)
        self.websocket_list.pop(user_idx)

        for user_id, websocket in zip(self.active_user_list,
                                      self.websocket_list):
            self.set_async_value(
                user_id=user_id,
                async_value={
                    'message': 'userCount',
                    'numUsers': self.num_users,
                },
                websocket=websocket,
            )

            logger.debug(f"SENT REMOVE USER NUMBER {self.num_users}")