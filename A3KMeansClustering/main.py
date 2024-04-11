from model import KMeans
from utils import get_image, show_image, save_image, error


def main():
    # get image
    image = get_image('image.jpg')
    img_shape = image.shape

    # reshape image
    image = image.reshape(image.shape[0] * image.shape[1], image.shape[2])

    # create model
    num_clusters = 50 # CHANGE THIS
    kmeans = KMeans(num_clusters)

    # fit model
    kmeans.fit(image)

    # replace each pixel with its closest cluster center
    image = kmeans.replace_with_cluster_centers(image)

    # reshape image
    image_clustered = image.reshape(img_shape)

    # Print the error
    print('MSE:', error(image, image_clustered))

    # show/save image
    # show_image(image)
    save_image(image_clustered, f'image_clustered_{num_clusters}.jpg')



if __name__ == '__main__':
    main()
