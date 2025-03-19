from kuwahara.kuwahara import image

filtered_image = image.apply_kuwahara_filter()
image._save_image(image_array=filtered_image)

