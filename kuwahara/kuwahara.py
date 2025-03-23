from kuwahara.kuwahara_image import KuwaharaFilterImage

def create_kuwahara_image(image_name, image_mode, kernel_size):
    return KuwaharaFilterImage(
        image_name=image_name,
        image_mode=image_mode,
        kernel_size=kernel_size,
    )