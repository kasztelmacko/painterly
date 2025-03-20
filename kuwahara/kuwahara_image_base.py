import numpy as np
import math
import cv2
from image_base import ImageBase
from typing import Union

class KuwaharaFilterImage(ImageBase):
    def __init__(self, image_name: str, image_mode: str, kernel_size: int = 5, n_subregions: int = 4):
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
        
        print("Padded image shape")
        return image_array

    def precompute_variance_lookup(self, padded_image: np.array = None):
        if padded_image is None:
            padded_image = self.padded_image

        height, width, channels = padded_image.shape
        variance_lookup = {
            "top_left": np.zeros((height, width, channels)),
            "top_right": np.zeros((height, width, channels)),
            "bottom_left": np.zeros((height, width, channels)),
            "bottom_right": np.zeros((height, width, channels))
        }

        for y in range(self.padding, height - self.padding):
            for x in range(self.padding, width - self.padding):
                for c in range(channels):
                    neighborhood = padded_image[
                        y - self.padding : y + self.padding + 1,
                        x - self.padding : x + self.padding + 1,
                        c
                    ]
                    quadrants = self.get_quadrants(neighborhood)
                    for quadrant_name, quadrant in quadrants.items():
                        variance_lookup[quadrant_name][y, x, c] = np.var(quadrant)

        print("Variance lookup table")
        return variance_lookup
    
    def apply_kuwahara_filter(self, image_array: np.array = None, padded_image: np.array = None):
        if image_array is None:
            image_array = self.image
        if padded_image is None:
            padded_image = self.padded_image
        
        height, width, channels = image_array.shape
        filtered_image = np.zeros_like(image_array)

        for y in range(height - self.padding):
            for x in range(width - self.padding):
                for c in range(channels):
                    neighborhood = self.get_neighbors(x, y, padded_image[:, :, c])
                    quadrants = self.get_quadrants(neighborhood)
                    best_mean = None
                    smallest_variance = float('inf')
                    
                    for quadrant_name, quadrant in quadrants.items():
                        mean = np.mean(quadrant)
                        variance = self.variance_lookup[quadrant_name][y, x, c]

                        if variance < smallest_variance:
                            smallest_variance = variance
                            best_mean = mean

                    if best_mean is None:
                        best_mean = image_array[y, x, c]

                    filtered_image[y, x, c] = best_mean

        filtered_image_rgb = filtered_image[:, :, ::-1]
        return filtered_image_rgb


    def get_neighbors(self, x: int, y: int, padded_image: np.array = None):
        if padded_image is None:
            padded_image = self.padded_image

        neighborhood = padded_image[
            y - self.padding : y + self.padding + 1,
            x - self.padding : x + self.padding + 1
        ]

        return neighborhood
    
    def get_quadrants(self, neighborhood: np.array):
        center = self.padding
        quadrants = {
            "top_left": neighborhood[:center + 1, :center + 1],
            "top_right": neighborhood[:center + 1, center:],
            "bottom_left": neighborhood[center:, :center +1 ],
            "bottom_right": neighborhood[center:, center:]
        }

        return quadrants
            
        


                

        
