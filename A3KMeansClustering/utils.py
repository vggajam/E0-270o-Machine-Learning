from matplotlib import pyplot as plt
import numpy as np

def get_image(image_path):
    image = plt.imread(image_path)
    return image/255.0


def show_image(image):
    plt.imshow(image)
    plt.show()

def save_image(image, image_path):
    plt.imsave(image_path, image)


def error(original_image: np.ndarray, clustered_image: np.ndarray) -> float:
    # Returns the Mean Squared Error between the original image and the clustered image
    return np.linalg.norm(original_image.reshape(-1)-clustered_image.reshape(-1))

def plot_and_save_k_vs_mse(xpoints, ypoints):
    plt.plot(xpoints, ypoints, marker="x")
    plt.title("K vs MSE")
    plt.xlabel("number of clusters ( K )")
    plt.ylabel("MSE")
    plt.savefig("k_vs_mse.png")
    plt.show()
