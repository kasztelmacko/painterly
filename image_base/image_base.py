import numpy as np
import cv2
from config import config

class ImageBase:
    def __init__(self, image_name, image_mode="BGR"):
        self.image_name = image_name
        self.image_path = config["images_path"] + image_name
        self.image_output_path = config["images_output_path"] + image_name
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

        self.image = self._resize_image_if_large(self.image)

        self.image_shape = self.image.shape

    def _resize_image_if_large(self, image_array: np.array, max_width: int = config["max_image_width"]):
        """
        Resize the image if its width exceeds the specified maximum width.
        The aspect ratio is maintained.
        """
        height, width = image_array.shape[:2]
        if width > max_width:
            new_width = max_width
            new_height = int((height / width) * new_width)
            image_array = cv2.resize(image_array, (new_width, new_height), interpolation=cv2.INTER_AREA)
            print(f"Image resized to {new_width}x{new_height} (original: {width}x{height})")
        return image_array

    def _show_image(self, image_array: np.array = None, image_name: str = None, window_size: tuple = None, window_position: tuple = None):
        if image_array is None:
            image_array = self.image
        if image_name is None:
            image_name = self.image_name

        cv2.namedWindow(image_name, cv2.WINDOW_NORMAL)
        if window_size:
            cv2.resizeWindow(image_name, window_size[0], window_size[1])
        if window_position:
            cv2.moveWindow(image_name, window_position[0], window_position[1])

        cv2.imshow(image_name, image_array)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def _save_image(self, filter_name, image_array: np.array = None, image_name: str = None):
        if image_array is None:
            image_array = self.image
        if image_name is None:
            image_name = self.image_output_path
        output_name = f"{image_name.split('.')[0]}_{filter_name}.{self.image_extension}"
        cv2.imwrite(output_name, image_array)

    def _validate_image_extension(self):
        if self.image_extension not in config["valid_image_extensions"]:
            raise Exception("Invalid image extension")
        
    def _normalize_image(self, image_array: np.array = None, scale: int = 255):
        if image_array is None:
            image_array = self.image
        if scale not in [1, 255]:
            raise ValueError("Scale must be either 1 or 255")
        return cv2.normalize(image_array, None, 0, scale, cv2.NORM_MINMAX)
    
    def _resize_image(self, image_array: np.array = None, resizer: float = 0.5):
        if image_array is None:
            image_array = self.image

        height, width = image_array.shape[:2]
        new_width = int(width * resizer)
        new_height = int(height * resizer)

        return cv2.resize(image_array, (new_width, new_height), interpolation=cv2.INTER_AREA)