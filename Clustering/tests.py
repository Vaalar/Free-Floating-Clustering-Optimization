import threading as th
import time

def positive(x):
    return x <= 0

print(min(filter(lambda x : x >= 0, [-1, 2, 0.0, 1])))

