from kuwahara.kuwahara_image_base import KuwaharaFilterImage

image = KuwaharaFilterImage(
    image_name="marti_test.jpg",
    image_mode="BGR"
)

filtered_image = image.apply_kuwahara_filter()
