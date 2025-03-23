from kuwahara.kuwahara import create_kuwahara_image
from kuwahara_anisotropic.kuwahara_anistoripic import create_anisotropic_kuwahara_image

def main(filter_type, image_name, image_mode, kernel_size, alpha=None, sharpness=None):
    if filter_type == "kuwahara":
        image = create_kuwahara_image(image_name, image_mode, kernel_size)
        filtered_image = image.apply_kuwahara_filter()
        image.save_filtered_image(filtered_image)
    elif filter_type == "anisotropic_kuwahara":
        image = create_anisotropic_kuwahara_image(image_name, image_mode, kernel_size, alpha, sharpness)
        filtered_image = image.apply_anisotropic_kuwahara_filter()
        image.save_filtered_image(filtered_image)
    else:
        raise ValueError("Invalid filter type. Choose 'kuwahara' or 'anisotropic_kuwahara'.")


if __name__ == "__main__":
    filter_type = "kuwahara"
    image_name = "Lenna.png"
    image_mode = "BGR"
    kernel_size = 13
    alpha = 1.0 if filter_type == "anisotropic_kuwahara" else None
    sharpness = 4.0 if filter_type == "anisotropic_kuwahara" else None

    main(filter_type, image_name, image_mode, kernel_size, alpha, sharpness)