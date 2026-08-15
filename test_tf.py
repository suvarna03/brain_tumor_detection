import time

start = time.time()

import tensorflow as tf

print(tf.__version__)

print(
    "Loaded in:",
    time.time() - start,
    "seconds"
)