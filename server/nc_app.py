"""
Simple wrapper around running a notebook + webserver backend
"""
import os

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import asyncio
from threading import Thread
import webserver
loop = asyncio.get_event_loop()

def mm_runner():
    webserver.main()
    loop.run_forever()
    
mm_thread = Thread(target=mm_runner, daemon=True)
mm_thread.start()

print("Kicking off notebook")
import vqgan  # this is blocking!