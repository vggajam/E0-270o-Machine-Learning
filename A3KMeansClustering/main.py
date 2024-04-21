from model import KMeans
from utils import get_image, show_image, save_image, error, plot_and_save_k_vs_mse


def main():
    # get image
    image = get_image('image.jpg')
    img_shape = image.shape

    # reshape image
    image = image.reshape(image.shape[0] * image.shape[1], image.shape[2])

    k_vals = [2,5,10,20,50]
    mse_vals = []
    for num_clusters in k_vals:
        # create model
        kmeans = KMeans(num_clusters)

        # fit model
        kmeans.fit(image)

        # replace each pixel with its closest cluster center
        image_clustered = kmeans.replace_with_cluster_centers(image)

        # reshape image
        image_clustered = image_clustered.reshape(img_shape)

        # Print the error
        mse = error(image, image_clustered)
        mse_vals.append(mse)
        print(f'k={num_clusters}; MSE: {mse}')

        # show/save image
        # show_image(image)
        save_image(image_clustered, f'image_clustered_{num_clusters}.jpg')

    plot_and_save_k_vs_mse(k_vals, mse_vals)


if __name__ == '__main__':
    main()
