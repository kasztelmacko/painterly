from kuwahara.kuwahara import create_kuwahara_image
from kuwahara_anisotropic.kuwahara_anistoripic import create_anisotropic_kuwahara_image
from kmeans.kmeans import create_kmeans_image

def main(filter_type, image_name, image_mode, kernel_size, n_clusters=None, alpha=None, sharpness=None):
    if filter_type == "kuwahara":
        image = create_kuwahara_image(image_name, image_mode, kernel_size)
        filtered_image = image.apply_kuwahara_filter()
        image.save_filtered_image(filtered_image)
    elif filter_type == "anisotropic_kuwahara":
        image = create_anisotropic_kuwahara_image(image_name, image_mode, kernel_size, alpha, sharpness)
        filtered_image = image.apply_anisotropic_kuwahara_filter()
        image.save_filtered_image(filtered_image)
    elif filter_type == "kmeans":
        image = create_kmeans_image(image_name, image_mode, n_clusters=15)
        filtered_image = image.apply_kmeans_filter()
        image.save_filtered_image(filtered_image)
    else:
        raise ValueError("Invalid filter type. Choose 'kuwahara' or 'anisotropic_kuwahara'.")


if __name__ == "__main__":
    filter_type = "kmeans"
    image_name = "Lenna.png"
    image_mode = "BGR"
    kernel_size = 13 if filter_type != "kmeans" else None
    n_clusters = 2 if filter_type == "kmeans" else None
    alpha = 1.0 if filter_type == "anisotropic_kuwahara" else None
    sharpness = 4.0 if filter_type == "anisotropic_kuwahara" else None

    main(filter_type, image_name, image_mode, kernel_size, alpha, sharpness, n_clusters)