import numpy as np
import math
import cv2
from image_base import ImageBase
from typing import Union

class KuwaharaFilterImage(ImageBase):
    def __init__(self, image_name: str, image_mode: str,kernel_size: int = 5, n_subregions: int = 4):
        super().__init__(
            image_name=image_name,
            image_mode=image_mode
        )
        self.kernel_size = kernel_size
        self.n_subregions = n_subregions
        self.padding = math.floor(kernel_size / 2)
        self.padded_image = self.pad_image()
        self.variance_lookup = self.precompute_variance_lookup()
        
    def pad_image(self, image_array: np.array = None, pad_type: str = "mirror"):
        if image_array is None:
            image_array = self.image
        if pad_type == "mirror":
            image_array = np.pad(image_array, self.padding, mode='reflect')
        elif pad_type == "zero":
            image_array = np.pad(image_array, self.padding, mode='constant')
        else:
            raise ValueError("pad_type must be either 'mirror' or 'zero'")
        return image_array

    def precompute_variance_lookup(self, padded_image: np.array = None):
        if padded_image is None:
            padded_image = self.padded_image

        height, width = padded_image.shape[:2]
        variance_lookup = np.zeros((height, width))

        for y in range(height):
            for x in range(width):
                subregion = padded_image[y:y+self.kernel_size, x:x+self.kernel_size]
                variance_lookup[y, x] = np.var(subregion)
        return variance_lookup
    
    def apply_kuwahara_filter(self, image_array: np.array = None, padded_image: np.array = None):
        if image_array is None:
            image_array = self.image
        if padded_image is None:
            padded_image = self.padded_image
        
        height, width = image_array.shape[:2]

        for y in range(height - self.padding - 1):
            for x in range(width - self.padding - 1):
                neighborhood = self.get_neighbors(x, y, padded_image)

    def get_neighbors(self, x: int, y: int, value: Union[int, tuple], padded_image: np.array = None):
        if padded_image is None:
            padded_image = self.padded_image

        neighborhood = padded_image[
            x - self.padding : x + self.padding + 1,
            y - self.padding : y + self.padding + 1
        ]

        return neighborhood
        


                

        
