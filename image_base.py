import numpy as np
import scipy as sp
import cv2
from config import config

class ImageBase:
    def __init__(self, image_path, image_name):
        self.image_name = image_name
        self.image_path = config["images_path"] + image_name
        self.image = cv2.imread(image_path)
        self.image_extension = image_name.split('.')[-1]

    def _show_image(self, image_array: np.array, image_name: str):
        cv2.imshow(image_name, image_array)
        cv2.waitKey(0)
    
    def _save_image(self, image_array: np.array, image_name: str):
        cv2.imwrite(image_name, image_array)

    def _validate_image_extension(self):
        if self.image_extension not in config["valid_image_extensions"]:
            raise Exception("Invalid image extension")