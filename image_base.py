import numpy as np
import scipy as sp
import cv2
from config import config

class ImageBase:
    def __init__(self, image_name, image_mode="BGR"):
        self.image_name = image_name
        self.image_path = config["images_path"] + image_name
        self.image_mode = image_mode
        self.image_extension = image_name.split('.')[-1]

        if image_mode == "greyscale":
            self.image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
            self.image_color_mode = "greyscale"
        elif image_mode == "RGB":
            self.image = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.image_color_mode = "RGB"
        elif image_mode == "BGR":
            self.image = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
            self.image_color_mode = "BGR"
        else:
            raise ValueError("Invalid image mode. Choose from 'greyscale', 'RGB', 'BGR'.")

        self.image_shape = self.image.shape

    def _show_image(self, image_array: np.array, image_name: str):
        cv2.imshow(image_name, image_array)
        cv2.waitKey(0)
    
    def _save_image(self, image_array: np.array, image_name: str):
        cv2.imwrite(image_name, image_array)

    def _validate_image_extension(self):
        if self.image_extension not in config["valid_image_extensions"]:
            raise Exception("Invalid image extension")
        
    def _normalize_image(self, image_array: np.array, scale: int = 255):
        if scale not in [1, 255]:
            raise ValueError("Scale must be either 1 or 255")
        return cv2.normalize(image_array, None, 0, scale, cv2.NORM_MINMAX)