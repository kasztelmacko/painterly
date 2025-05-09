import numpy as np
import cv2
import matplotlib.pyplot as plt

def normal(value, max_value):
    return int(np.clip(value * max_value, 0, max_value - 1))

def draw(f, width=128):
    x0, y0, x1, y1, x2, y2, z0, z2, w0, w2 = f
    x1 = x0 + (x2 - x0) * x1
    y1 = y0 + (y2 - y0) * y1
    x0 = normal(x0, width * 2)
    x1 = normal(x1, width * 2)
    x2 = normal(x2, width * 2)
    y0 = normal(y0, width * 2)
    y1 = normal(y1, width * 2)
    y2 = normal(y2, width * 2)
    z0 = int(1 + z0 * width // 4)
    z2 = int(1 + z2 * width // 4)
    
    canvas = np.zeros([width * 2, width * 2], dtype=np.float32)
    
    tmp = 1.0 / 100
    for i in range(100):
        t = i * tmp
        x = int((1 - t) ** 2 * x0 + 2 * t * (1 - t) * x1 + t ** 2 * x2)
        y = int((1 - t) ** 2 * y0 + 2 * t * (1 - t) * y1 + t ** 2 * y2)
        z = int((1 - t) * z0 + t * z2)
        w = (1 - t) * w0 + t * w2
        cv2.circle(canvas, (y, x), z, w, -1)
    
    return 1 - cv2.resize(canvas, dsize=(width, width))