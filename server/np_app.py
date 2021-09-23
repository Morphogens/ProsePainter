"""
Simple wrapper around running a notebook + webserver backend
"""
import os

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
# input()
import vqgan  # this is blocking!