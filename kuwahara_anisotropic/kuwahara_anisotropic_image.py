import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
from image_base import ImageBase

class AnisotropicKuwaharaImage(ImageBase):
    def __init__(self, image_name: str, image_mode: str, kernel_size: int = 5, alpha: float = 1.0, sharpness: float = 4.0):
        super().__init__(
            image_name=image_name,
            image_mode=image_mode
        )
        self.kernel_size = kernel_size
        self.alpha = alpha
        self.sharpness = sharpness 
        self.padding = (kernel_size - 1) // 2
        self.padded_image = self.pad_image()

    def pad_image(self, image_array: np.array = None, pad_type: str = "mirror"):
        if image_array is None:
            image_array = self.image
        if pad_type == "mirror":
            image_array = np.pad(image_array, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='reflect')
        elif pad_type == "zero":
            image_array = np.pad(image_array, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')
        else:
            raise ValueError("pad_type must be either 'mirror' or 'zero'")
        
        return image_array

    def apply_anisotropic_kuwahara_filter(self, image_array: np.array = None):
        if image_array is None:
            image_array = self.image
        
        if len(image_array.shape) == 2:
            filtered_image = self.AnisotropicKuwahara(image_array, self.kernel_size, self.alpha, self.sharpness)
        elif len(image_array.shape) == 3:
            filtered_channels = []
            for channel in range(image_array.shape[2]):
                filtered_channel = self.AnisotropicKuwahara(image_array[:, :, channel], self.kernel_size, self.alpha, self.sharpness)
                filtered_channels.append(filtered_channel)
            filtered_image = np.stack(filtered_channels, axis=2)
        else:
            raise ValueError("Unsupported image format. Image must be grayscale or color (2D or 3D array).")
        
        return filtered_image

    def AnisotropicKuwahara(self, image_array, kernel_size, alpha, sharpness):
        image = image_array.astype(np.float64)
        if kernel_size % 4 != 1:
            raise Exception("Invalid kernel_size %s: kernel_size must follow formula: w = 4*n+1." % kernel_size)

        grad_x = convolve2d(image, np.array([[-1, 0, 1]]), mode='same')
        grad_y = convolve2d(image, np.array([[-1], [0], [1]]), mode='same')
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        grad_orientation = np.arctan2(grad_y, grad_x)

        grad_magnitude = gaussian_filter(grad_magnitude, sigma=1)
        grad_orientation = gaussian_filter(grad_orientation, sigma=1)

        grad_magnitude = grad_magnitude / np.max(grad_magnitude)

        half_size = (kernel_size - 1) // 2
        y, x = np.mgrid[-half_size:half_size+1, -half_size:half_size+1]
        kernel_shape = (kernel_size, kernel_size)

        filtered = np.zeros_like(image)
        for row in range(image.shape[0]):
            for col in range(image.shape[1]):
                theta = grad_orientation[row, col]
                magnitude = grad_magnitude[row, col]

                adjusted_alpha = alpha * (1 + magnitude * sharpness)

                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)
                x_rot = x * cos_theta - y * sin_theta
                y_rot = x * sin_theta + y * cos_theta

                kernel = np.exp(-(x_rot**2 + (adjusted_alpha * y_rot)**2) / (2 * (half_size / 2)**2))
                kernel = kernel / np.sum(kernel)

                local_region = image[row-half_size:row+half_size+1, col-half_size:col+half_size+1]
                if local_region.shape == kernel_shape:
                    filtered[row, col] = np.sum(local_region * kernel)

        return filtered.astype(np.uint8)
    
    def save_filtered_image(self, filtered_image):
        self._save_image("anisotropic_kuwahara", image_array=filtered_image)