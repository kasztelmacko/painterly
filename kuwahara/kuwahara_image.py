import numpy as np
from scipy.signal import convolve2d
from image_base import ImageBase

class KuwaharaFilterImage(ImageBase):
    def __init__(self, image_name: str, image_mode: str, kernel_size: int = 5):
        super().__init__(
            image_name=image_name,
            image_mode=image_mode
        )
        self.kernel_size = kernel_size
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

    def apply_kuwahara_filter(self, image_array: np.array = None):
        if image_array is None:
            image_array = self.image
        
        if len(image_array.shape) == 2:
            filtered_image = self.Kuwahara(image_array, self.kernel_size)
        elif len(image_array.shape) == 3:
            filtered_channels = []
            for channel in range(image_array.shape[2]):
                filtered_channel = self.Kuwahara(image_array[:, :, channel], self.kernel_size)
                filtered_channels.append(filtered_channel)
            filtered_image = np.stack(filtered_channels, axis=2)
        else:
            raise ValueError("Unsupported image format. Image must be grayscale or color (2D or 3D array).")
        
        return filtered_image

    def Kuwahara(self, image_array, kernel_size):
        image = image_array.astype(np.float64)
        if kernel_size % 4 != 1:
            raise Exception("Invalid kernel_size %s: kernel_size must follow formula: w = 4*n+1." % kernel_size)

        half_size = (kernel_size - 1) // 2
        tmpAvgKerRow = np.hstack((np.ones((1, half_size + 1)), np.zeros((1, half_size))))
        tmpPadder = np.zeros((1, kernel_size))
        tmpavgker = np.tile(tmpAvgKerRow, (half_size + 1, 1))
        tmpavgker = np.vstack((tmpavgker, np.tile(tmpPadder, (half_size, 1))))
        tmpavgker = tmpavgker / np.sum(tmpavgker)

        avgker = np.empty((4, kernel_size, kernel_size)) 
        avgker[0] = tmpavgker			    # top-left (a)
        avgker[1] = np.fliplr(tmpavgker)	# top-right (b)
        avgker[2] = np.flipud(tmpavgker)	# bottom-left (c)
        avgker[3] = np.fliplr(avgker[2])	# bottom-right (d)
        
        squaredImg = image**2
        
        avgs = np.zeros([4, image.shape[0], image.shape[1]])
        stddevs = avgs.copy()

        for k in range(4):
            avgs[k] = convolve2d(image, avgker[k], mode='same')
            stddevs[k] = convolve2d(squaredImg, avgker[k], mode='same')
            stddevs[k] = stddevs[k] - avgs[k]**2
        
        indices = np.argmin(stddevs, 0)

        filtered = np.zeros(image_array.shape)
        for row in range(image_array.shape[0]):
            for col in range(image_array.shape[1]):
                filtered[row, col] = avgs[indices[row, col], row, col]

        return filtered.astype(np.uint8)
    
    def save_filtered_image(self, filtered_image):
        self._save_image("kuwahara", image_array=filtered_image)