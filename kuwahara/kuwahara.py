import cv2
import numpy as np
from image_base import ImageBase

image = ImageBase(
    image_name="marti_test.jpg",
    image_mode="greyscale"
)

image._show_image()