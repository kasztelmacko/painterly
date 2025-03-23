from kuwahara_anisotropic.kuwahara_anisotropic_image import AnisotropicKuwaharaImage

def create_anisotropic_kuwahara_image(image_name, image_mode, kernel_size, alpha, sharpness):
    return AnisotropicKuwaharaImage(
        image_name=image_name,
        image_mode=image_mode,
        kernel_size=kernel_size,
        alpha=alpha,
        sharpness=sharpness,
    )