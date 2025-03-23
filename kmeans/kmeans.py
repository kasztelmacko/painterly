from kmeans.kmeans_image import KMeansFilterImage

def create_kmeans_image(image_name, image_mode, n_clusters):
    return KMeansFilterImage(
        image_name=image_name,
        image_mode=image_mode,
        n_clusters=n_clusters,
   )