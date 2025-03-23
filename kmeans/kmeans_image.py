from image_base.image_base import ImageBase
import numpy as np
import torch
from fast_pytorch_kmeans import KMeans

class KMeansFilterImage(ImageBase):
    def __init__(self, image_name: str, image_mode: str, n_clusters: int = 14):
        super().__init__(
            image_name=image_name,
            image_mode=image_mode
        )
        self.n_clusters = n_clusters

    def apply_kmeans_filter(self, image_array: np.array = None):
        if image_array is None:
            image_array = self.image
        
        if len(image_array.shape) == 2:
            pixels = image_array.reshape(-1, 1)
        elif len(image_array.shape) == 3:
            pixels = image_array.reshape(-1, image_array.shape[2])
        else:
            raise ValueError("Unsupported image format. Image must be grayscale or color (2D or 3D array).")

        pixels_tensor = torch.tensor(pixels, dtype=torch.float32)
        kmeans = KMeans(n_clusters=self.n_clusters, mode='euclidean')
        labels = kmeans.fit_predict(pixels_tensor)

        cluster_centers = kmeans.centroids
        filtered_pixels = cluster_centers[labels].cpu().numpy()

        if len(image_array.shape) == 2:
            filtered_image = filtered_pixels.reshape(image_array.shape)
        else:
            filtered_image = filtered_pixels.reshape(image_array.shape)
        
        return filtered_image.astype(np.uint8)
    
    def save_filtered_image(self, filtered_image):
        self._save_image("kmeans", image_array=filtered_image)